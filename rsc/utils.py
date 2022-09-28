from itertools import product
import pandas as pd
import numpy as np


def get_cell_parameter(cif_file):
    cell_length = []
    cell_angle = []
    with open(cif_file, 'r') as f:
        lines = f.read().split('\n')

    length_tags = [
        '_cell_length_a', '_cell_length_b', '_cell_length_c'
    ]
    angle_tags = [
        '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'
    ]

    # get unit cell lengths and angles
    for line, len_tag in product(lines, length_tags):
        if line.startswith(len_tag):
            cell_length.append(float(line.split()[-1].split('(')[0]))
    for line, ang_tag in product(lines, angle_tags):
        if line.startswith(ang_tag):
            cell_angle.append(float(line.split()[-1].split('(')[0]))

    return cell_length, cell_angle


def A(a):  # return the unit cell vector A
    return np.array([a, 0, 0])


def B(b, gamma):  # return the unit cell vector B
    return np.array([b * np.cos(gamma), b * np.sin(gamma), 0])


def C(c, alpha, beta, gamma):  # return the unit cell vector C
    return np.array([
        c * np.cos(beta),
        c * np.cos(alpha) - np.cos(beta) * np.cos(gamma),
        c * np.sqrt(
                1 - np.cos(beta)**2 -
                ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)))
                / (np.sin(gamma))
            )**2
    ])


def get_lat_mat(cell_length, cell_angle):
    # return the matrix of the unit cell from the cell lengths and angles
    a, b, c = cell_length
    alpha, beta, gamma = [np.deg2rad(ang) for ang in cell_angle]
    lat_m = np.array([A(a), B(b, gamma), C(c, alpha, beta, gamma)])
    lat_m = np.round(lat_m, 6)
    return lat_m


def get_sym_constraints(cif_file):
    with open(cif_file, 'r') as f:
        lines = f.read().split('\n')
    xyz_index = [i for i, l in enumerate(lines) if l.startswith('loop')]
    xyz_slice = slice(xyz_index[0]+2, xyz_index[1])
    xyz_lines = lines[xyz_slice]
    # clean up the xyz lines remove tabs, commas, and quotes
    for ch in [' ', '\t', "'"]:
        xyz_lines = [line.replace(ch, '') for line in xyz_lines]
    transforms = [lambda x, y, z, coef=i: eval(f'[{coef}]') for i in xyz_lines]
    return transforms


def parse_atom_site(cif_file):
    with open(cif_file, 'r') as f:
        lines = f.read().split('\n')
    site_id = [
        id for id, line in enumerate(lines) if line.startswith('_atom_site_')
    ]
    site_slice = slice(site_id[0], site_id[-1]+1)
    col_names = [line.split('_atom_site_')[-1] for line in lines[site_slice]]
    data = [line.split() for line in lines[site_id[-1]+1:]]
    df = pd.DataFrame(data, columns=col_names)
    df['site'] = [lab[len(ts):] for ts, lab in zip(df.type_symbol, df.label)]
    df.occupancy = [float(i.split('(')[0]) for i in df.occupancy]
    df.symmetry_multiplicity = df.symmetry_multiplicity.astype(int)
    return df


def gen_site_pos(df, transforms):
    sites = df.site.unique()
    site_frac = {}
    for site in sites:
        frac_names = ['fract_x', 'fract_y', 'fract_z']
        x, y, z = df.loc[df.site == site][frac_names].values[0]
        x, y, z = [float(i.split('(')[0]) for i in [x, y, z]]
        frac = np.unique([t(x, y, z) for t in transforms], axis=0)
        frac = frac % 1
        frac = np.unique(frac, axis=0)
        site_frac[site] = frac
    return site_frac


def extend_cell(site_frac, n):
    xyz = [1, 0, 0]
    for site, pos in site_frac.items():
        for _ in range(3):
            # here we could implement a anisotropic way to extend the cell
            x, y, z = xyz
            dim = pos.shape[0]
            ext = np.vstack([
                x * np.ones((dim)),
                y * np.ones((dim)),
                z * np.ones((dim))
            ]).T
            pos = np.vstack([pos, pos + ext])
            xyz = np.roll(xyz, 1)
        site_frac[site] = pos
    return site_frac


def coord_fully_occupied_single_site(
            site_occ, site_com, site_frac, df
        ):
    poscar_frac = []
    for site in site_occ.keys():
        occupancy_sum = df[df.site == site].occupancy.sum()
        if site_occ[site] == 1 and occupancy_sum == 1:
            site_com[site] = True
            atom = df[df.site == site].type_symbol.to_list()[0]
            poscar_frac.append([[atom, *frac]for frac in site_frac[site]])

    return site_com, poscar_frac


def random_shared_site_occupation(site_occ, site_com, site_frac, df):
    poscar_frac = []
    for site, occ in site_occ.items():
        if site_occ[site] > 1:
            site_com[site] = True
            cols = ['type_symbol', 'occupancy']
            atom, occu = df[df.site == site][cols].to_numpy().T
            dim = len(site_frac[site])
            assigment = [
                np.repeat(a, round(o*dim)) for a, o in zip(atom, occu)
                ]
            assigment = np.concatenate(assigment)
            np.random.shuffle(assigment)
            poscar_frac.append(
                [[a, *frac]for a, frac in zip(assigment, site_frac[site])]
                )

            COUNTS = {a: 0 for a in atom}
            for v in assigment:
                for a in atom:
                    if v == a:
                        COUNTS[a] += 1
    return site_com, poscar_frac


def under_occupied_site(site_occ, site_com, site_frac, df):
    poscar_frac = []
    for site, occ in site_occ.items():
        if occ == 1 and df[df.site == site].occupancy.sum() < 1:
            site_com[site] = True
            atom = df[df.site == site].type_symbol.to_list()[0]
            number_atoms = site_frac[site].shape[0]
            prop = df[df.site == site].occupancy.sum()
            prop = np.array([prop, 1-prop])
            print(
                f'Atoms: {atom} site {site} is under occupied',
                df[df.site == site].occupancy.sum()
            )
            atom = np.array([atom, 'empty'])
            assigment = [
                np.repeat(a, round(o*number_atoms)) for a, o in zip(atom, prop)
                ]
            assigment = np.concatenate(assigment)
            np.random.shuffle(assigment)
            atom = atom[0]
            poscar_frac = list(poscar_frac)
            poscar_frac.append(
                [[a, *frac]for a, frac in zip(assigment, site_frac[site])]
                )
            poscar_frac = np.vstack(poscar_frac)
            poscar_frac = poscar_frac[poscar_frac[:, 0] != 'empty']
    return site_com, poscar_frac


def write_POSCAR(out_dir, cif, df, POSCAR_positions, COUNTS, ATOMS, lat_mat):
    name = cif.split('/')[-1].split('.')[0]
    comment = name
    for label, occu in zip(df.label, df.occupancy):
        comment += f'__{label}_{occu}__'
    with open(f'{out_dir}POSCAR_{name}.vasp', 'w') as f:
        for atom in ATOMS:
            f.write(f'{comment} ')
        f.write('\n1.0\n')
        for i in lat_mat:
            f.write(f'{i[0]:.6f} {i[1]:.6f} {i[2]:.6f}\n')
        for atom in ATOMS:
            f.write(f'{atom} ')
        f.write('\n')
        for atom in ATOMS:
            f.write(f'{COUNTS[atom]} ')
        f.write('\n')
        f.write('Direct\n')
        for i in POSCAR_positions[:-1]:
            i = [float(j) for j in i[1:]]
            f.write(f'{i[0]:.6f} {i[1]:.6f} {i[2]:.6f}\n')
        i = POSCAR_positions[-1]
        i = [float(j) for j in i[1:]]
        f.write(f'{i[0]:.6f} {i[1]:.6f} {i[2]:.6f}')

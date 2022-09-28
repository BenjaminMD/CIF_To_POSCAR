import rsc.utils as ut
import numpy as np


class CIF_to_POSCAR():
    """
    read cif file and create poscar file
    the df file allows manual changes to the poscar file
    its advisable to move the utility functions to class to reduce args passed
    """
    def __init__(self, cif, out_dir):
        self.cif = cif
        self.out_dir = out_dir
        self.df = ut.parse_atom_site(cif)
        self.transforms = ut.get_sym_constraints(cif)
        self.cell_length, self.cell_angle = ut.get_cell_parameter(self.cif)
        self.lat_mat = ut.get_lat_mat(self.cell_length, self.cell_angle)
        self.site_frac = ut.gen_site_pos(self.df, self.transforms)
        self.no_atoms = self.df.symmetry_multiplicity.sum()
        self.POSCAR_positions = [] 

        if self.no_atoms < 50:  # condition  if cell is small
            print(f'No of atoms: {self.no_atoms}, less than 50 extending cell')
            self.df.symmetry_multiplicity = (
                self.df.symmetry_multiplicity * 2**3
            )
            self.site_frac = ut.extend_cell(self.site_frac, 1)
            self.supercell = 2
            self.lat_mat = self.lat_mat * 2
            print(f'No of atoms: {self.no_atoms* 2**3}\n-------------------')

    def write_POSCAR(self):
        site_com, POSCAR_positions = self.apply_occupations_vacancies()
        if all(site_com.values()):
            POSCAR_positions = np.vstack(POSCAR_positions)
            POSCAR_positions[POSCAR_positions[:, 0].argsort()]

            ATOMS = np.unique(POSCAR_positions[:, 0])
            COUNTS = {a: 0 for a in ATOMS}
            for v in POSCAR_positions[:, 0]:
                for a in ATOMS:
                    if v == a:
                        COUNTS[a] += 1
            print('____________Counts', COUNTS, '_________')
        else:
            print('Not all sites are complete')
        POSCAR_positions = POSCAR_positions[
            POSCAR_positions[:, 0].argsort()
        ]
        ut.write_POSCAR(
            self.out_dir,
            self.cif,
            self.df,
            POSCAR_positions,
            COUNTS,
            ATOMS,
            self.lat_mat
        )

    def apply_occupations_vacancies(self):
        df = self.df
        site_frac = self.site_frac
        POSCAR_positions = self.POSCAR_positions
        site_occ = {
            site: df[df.site == site].shape[0] for site in site_frac.keys()
        }
        site_com = {site: False for site in site_frac.keys()}
        for site, coordinates in site_frac.items():
            coordinates = coordinates / self.supercell
            site_frac[site] = coordinates
        site_com, poscar_frac = ut.coord_fully_occupied_single_site(
            site_occ, site_com, site_frac, df
        )
        POSCAR_positions.extend(poscar_frac)
        site_com, poscar_frac = ut.random_shared_site_occupation(
            site_occ, site_com, site_frac, df
        )
        POSCAR_positions.extend(poscar_frac)
        site_com, poscar_frac = ut.under_occupied_site(
            site_occ, site_com, site_frac, df
        )
        POSCAR_positions.extend(poscar_frac)

        return site_com, POSCAR_positions

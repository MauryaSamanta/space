class ExperimentConfig:
    def __init__(
        self,
        name,
        n_oos=1,
        n_satellites=3,
        n_orbits=1,
        fuel_per_oos=5.0,
        steps=1000,
        sensing_radius=5_000_000.0
    ):
        self.name = name
        self.n_oos = n_oos
        self.n_satellites = n_satellites
        self.n_orbits = n_orbits
        self.fuel_per_oos = fuel_per_oos
        self.steps = steps
        self.sensing_radius=sensing_radius
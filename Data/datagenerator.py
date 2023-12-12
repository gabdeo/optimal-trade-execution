import numpy as np


class DataGenerator:
    def __init__(self, config, n_samples):
        """
        Initialize the DataGenerator with simulation parameters provided in a dictionary.

        :param config: A dictionary with the following structure:
                       {
                           'T': int,
                           'volume': {'max': float, 'min': float, 'generation_type': str},
                           'quantity': {'max': float, 'min': float, 'generation_type': str},
                           'prices': {'vol': float, 'sensi': float, 'P0': float}
                       }
        :param n_samples: The number of vector samples to generate.
        The 'prices' configuration is optional and has a default value for 'P0'.
        """
        self.T = config.get("T", None)
        self.n_samples = n_samples
        self.volume_config = config.get("volume", None)
        self.quantity_config = config.get("quantity", None)
        self.prices_config = config.get(
            "prices", {"vol": None, "sensi": None, "P0": 1.0}
        )

    def generate_volumes(self):
        if self.volume_config:
            if self.volume_config["generation_type"] == "random":
                return np.random.uniform(
                    self.volume_config["min"],
                    self.volume_config["max"],
                    (self.n_samples, self.T),
                )
            elif self.volume_config["generation_type"] == "equal":
                return np.full(
                    (self.n_samples, self.T),
                    (self.volume_config["max"] + self.volume_config["min"]) / 2,
                )
            elif self.volume_config["generation_type"] == "visu":
                volumes = np.linspace(
                    self.volume_config["min"], self.volume_config["max"], self.n_samples
                )
                return np.tile(volumes, (self.T, 1)).T
        return None

    def generate_quantities(self):
        if self.quantity_config:
            if self.quantity_config["generation_type"] == "random":
                return np.random.uniform(
                    self.quantity_config["min"],
                    self.quantity_config["max"],
                    (self.n_samples, self.T),
                )
            elif self.quantity_config["generation_type"] == "equal":
                return np.full(
                    (self.n_samples, self.T),
                    (self.quantity_config["max"] + self.quantity_config["min"]) / 2,
                )
            elif self.quantity_config["generation_type"] == "visu":
                quantities = np.linspace(
                    self.quantity_config["min"],
                    self.quantity_config["max"],
                    self.n_samples,
                )
                return np.tile(quantities, (self.T, 1)).T
        return None

    def generate_prices(self):
        if self.prices_config["vol"] and self.prices_config["sensi"]:
            P0 = self.prices_config["P0"]
            vol = self.prices_config["vol"]
            sensi = self.prices_config["sensi"]
            all_prices = np.zeros((self.n_samples, self.T))
            all_prices[:, 0] = P0
            for i in range(self.n_samples):
                for t in range(1, self.T):
                    zt = np.random.normal(0, 1)  # z_t ~ N(0, 1)
                    qt = np.random.uniform(
                        self.quantity_config["min"], self.quantity_config["max"]
                    )
                    vt = np.random.uniform(
                        self.volume_config["min"], self.volume_config["max"]
                    )
                    pt = sensi * np.sqrt(qt / vt) + vol * zt + all_prices[i, t - 1]
                    all_prices[i, t] = pt
            return all_prices
        return None

    def generate_data(self):
        """
        Generate quantities, volumes, and prices as configured.

        :return: A tuple of numpy arrays or None, for quantities, volumes, and prices.
        Each array has a shape of (n_samples, T) where n_samples is the number of vector samples generated.
        """
        quantities = self.generate_quantities()
        volumes = self.generate_volumes()
        prices = (
            self.generate_prices()
            if quantities is not None and volumes is not None
            else None
        )
        return quantities, volumes, prices


"""F450FlightEnv icin custom policy bilesenleri.

F450FlightEnv'in 15 boyutlu gozlemi (bkz. f450_flight_env.py _get_obs()):

    idx  0    : alt_err        (irtifa hatasi)
    idx  1    : hdot           (dikey hiz)
    idx  2    : along_n        (ileri hiz, hedef yone gore)
    idx  3    : cross_n        (yana kayma, hedef yone gore)
    idx  4    : roll
    idx  5    : pitch
    idx  6-8  : p, q, r        (acisal hizlar)
    idx  9-10 : sin(heading), cos(heading)
    idx 11-14 : onceki aksiyon (4 motor)

Varsayilan MlpPolicy bu 15 sayiyi tek bir duz katmana verir; ag hangi
sayinin neye karsilik geldigini sifirdan kesfetmek zorunda kalir.

FlightFeaturesExtractor bunun yerine gozlemi 4 fiziksel gruba ayirip
HER GRUBA AYRI kucuk bir MLP uygular, sonuclari birlestirip tek bir
ozellik vektorune indirger. Boylece agin "dikey durum" ve "yatay/yon
durumu" icin ayri, temiz bir ic temsili olur - bu da f450_flight_env.py
'deki progress-agirlikli odul seklini (once dikey, sonra yatay) ogrenmeyi
kolaylastirmasi beklenir.

Bu extractor hem actor (pi) hem critic (vf) tarafindan PAYLASILIR
(SB3'un varsayilan mimarisi boyle calisir); ayri pi/vf agirliklari
ise zaten train.py'deki net_arch=dict(pi=.., vf=..) ile saglaniyor.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


# F450FlightEnv._get_obs() ile BIREBIR ayni sirada olmali.
VERTICAL_IDX = [0, 1]           # alt_err, hdot
HORIZONTAL_IDX = [2, 3, 9, 10]  # along_n, cross_n, sin_h, cos_h
ATTITUDE_IDX = [4, 5, 6, 7, 8]  # roll, pitch, p, q, r
ACTION_IDX = [11, 12, 13, 14]   # onceki aksiyon (4 motor)


def _branch(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
        nn.Tanh(),
    )


class FlightFeaturesExtractor(BaseFeaturesExtractor):
    """F450FlightEnv'e ozel, gruplandirilmis-dal (grouped-branch) feature extractor.

    Parametreler
    ----------
    observation_space : gymnasium.spaces.Box, shape (15,) bekleniyor.
    features_dim : birlesik ciktinin boyutu (bundan sonra SB3'un kendi
        net_arch=[128,128] pi/vf katmanlarina girer).
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]
        if obs_dim != 15:
            raise ValueError(
                f"FlightFeaturesExtractor 15 boyutlu flight gozlemi bekliyor, "
                f"{obs_dim} geldi. (Bu extractor'i hover gorevinde kullanma.)"
            )

        # Indeksleri buffer olarak sakla ki .to(device) ile birlikte tasinsin.
        self.register_buffer("vertical_idx", torch.tensor(VERTICAL_IDX, dtype=torch.long))
        self.register_buffer("horizontal_idx", torch.tensor(HORIZONTAL_IDX, dtype=torch.long))
        self.register_buffer("attitude_idx", torch.tensor(ATTITUDE_IDX, dtype=torch.long))
        self.register_buffer("action_idx", torch.tensor(ACTION_IDX, dtype=torch.long))

        self.vertical_branch = _branch(len(VERTICAL_IDX), 32, 16)
        self.horizontal_branch = _branch(len(HORIZONTAL_IDX), 32, 16)
        self.attitude_branch = _branch(len(ATTITUDE_IDX), 32, 16)
        self.action_branch = _branch(len(ACTION_IDX), 16, 8)

        combined_dim = 16 + 16 + 16 + 8  # = 56
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        vertical = self.vertical_branch(observations.index_select(1, self.vertical_idx))
        horizontal = self.horizontal_branch(observations.index_select(1, self.horizontal_idx))
        attitude = self.attitude_branch(observations.index_select(1, self.attitude_idx))
        prev_action = self.action_branch(observations.index_select(1, self.action_idx))

        combined = torch.cat([vertical, horizontal, attitude, prev_action], dim=1)
        return self.combine(combined)


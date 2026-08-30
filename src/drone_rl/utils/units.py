"""Birim donusumleri. JSBSim emperyal birim kullanir."""

FT2M = 0.3048
M2FT = 1.0 / FT2M

def ft_to_m(x):
    return x * FT2M

def m_to_ft(x):
    return x * M2FT

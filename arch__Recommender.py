# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2023 Animo Omnis Corporation. All Rights Reserved.

Thank you for your curiosity!

Arch file for recommender
"""

import ao_arch as ar

description = "Basic Recommender System"

#genre, length
#arch_i = [int(5694/2)]



#arch_i = [64, 64, 64, 64, 64, 64, 10, 10, 12, 64, 10, 10, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10,  8,  4, 11, 11]
arch_i = [64, 64, 64, 64, 64, 64, 10, 10, 12, 64, 10, 10, 64, 64, 64,  8,  8, 8,  8,  8,  8,  8,  8,  8,  8, 64, 64, 64, 64, 64, 10,  8,  4, 11, 11]



arch_z = [10]           
arch_c = []           
connector_function = "full_conn"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ao.Arch" in the line below (the API is pre-loaded with a version of the Arch class in this repo's main branch, hence "ao.Arch")
arch = ar.Arch(arch_i, arch_z, arch_c, connector_function, description)


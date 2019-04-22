# -*- coding: utf-8 -*-

from preprocessing.iaps import iaps

iaps_class = iaps()
print(iaps_class.get_pic_id(0))
print(iaps_class.get_sample_idx(1050))
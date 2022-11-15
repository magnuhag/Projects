import numpy as np
import pandas as pd

"""
Assumtion 1 (A1):
    We can fit 5 vinyls in each box, a box having a thickness of 5
Assumtion 2 (A2):
    We can fit 5 cds in each letter

"""


class records:

    def __init__(self, n_twelve_inch_records = None, n_ten_inch_vinyl_records = None, n_cd_records = None):
        self.n_twelve_inch_vinyl_records = n_twelve_inch_records
        self.n_ten_inch_vinyl_records = n_ten_inch_vinyl_records
        self.n_cd_records = n_cd_records

        self.vinyl_stack_thickness = 0
        self.cd_stack_thickness = 0

        self.vinyl_weight = 0
        self.cd_weight = 0

    def twelve_inch(self):
        thickness = 1 #unit
        weight = 220 #grams
        self.vinyl_stack_thickness += self.n_twelve_inch_vinyl_records*thickness
        self.vinyl_weight += weight*self.n_twelve_inch_vinyl_records

    def ten_inch(self):
        thickness = 1
        weight = 180
        self.vinyl_stack_thickness += self.n_ten_inch_vinyl_records*thickness
        self.vinyl_weight += weight*self.n_ten_inch_vinyl_records

    def cds(self):
        thickness = 1
        weight = 70
        self.cd_stack_thickness += self.n_cd_records*thickness
        self.cd_weight += self.n_cd_records*weight


    def total(self):
        info_dict = {
        "n_twelve_inches": self.n_twelve_inch_vinyl_records,
        "n_ten_inches": self.n_ten_inch_vinyl_records,
        "n_cds": self.n_cd_records,
        "tot_weight_vinyl": self.vinyl_weight,
        "tot_vinyl_thickness": self.vinyl_weight,
        "tot_cd_weight": self.cd_weight}
        return info_dict

class shipment(records):



shipment = records(n_twelve_inch_records = 3 , n_ten_inch_vinyl_records = 5, n_cd_records = 4)
dic = shipment.total()
print(pd.DataFrame.from_dict(dic, orient = "index"))

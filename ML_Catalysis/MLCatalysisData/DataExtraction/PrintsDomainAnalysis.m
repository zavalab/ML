%{
    This script uses the input file specified to create an array ("Domain")
    that contains the min, max, mean and stdev after normalizing.
%}

clear all;
clc;

data_table = readtable('final_withoutPtAuCeO2_notNormalized.txt');

Domain = [];

Attribute = data_table.BE_C;
Domain = [Domain; ["BE_C",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.loading_base;
Domain = [Domain; ["loading_base",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.Z_IonicRad;
Domain = [Domain; ["Z_IonicRad",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.Electronegativity;
Domain = [Domain; ["BE_C",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.loading_promoter;
Domain = [Domain; ["Electronegativity",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.FirstIE_supp;
Domain = [Domain; ["FirstIE_supp",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.Electroneg_supp;
Domain = [Domain; ["Electroneg_supp",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.CalcT_C;
Domain = [Domain; ["CalcT_C",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.CalcT_time;
Domain = [Domain; ["CalcT_time",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.T_K;
Domain = [Domain; ["T_K",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.H2Vol_;
Domain = [Domain; ["H2Vol_",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.COVol_;
Domain = [Domain; ["COVol_",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.H2OVol_;
Domain = [Domain; ["H2OVol_",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.CO2Vol_;
Domain = [Domain; ["CO2Vol_",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.TOS_min_;
Domain = [Domain; ["TOS_min_",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.F_W;
Domain = [Domain; ["F_W",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];

Attribute = data_table.rate_for_adj_norm;
Domain = [Domain; ["rate_for_adj_norm",  min(Attribute), max(Attribute)], mean(Attribute), std(Attribute) ];




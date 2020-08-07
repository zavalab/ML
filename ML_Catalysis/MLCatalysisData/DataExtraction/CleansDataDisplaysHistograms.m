%{

This script will read in the original excel sheet, clean the data and
display the histograms. I would reccomend re-writing this script to read in
one of the txt files with cleaned data, to ensure that the histograms match
the final dataset. 

%}

clear all
clc

data_table = readtable('Odabasi_2014_supp.xlsx','Sheet','input4k');

% Remove these columns from the table
data_table.Total_OfData = [];
% data_table.Reference = [];
data_table.Data = [];
data_table.T_C = [];
data_table.W_F = [];
% data_table.F_W = [];
data_table.F_CO_W = [];
% data_table.CO_Conv = [];
data_table.mol_mL = [];
data_table.F_CO_W_molsCO_min_mgCat_ = [];
data_table.Rate_molsCO_min_mgCat_ = [];
data_table.gCat_molMetal = [];
% data_table.Rate_molsCO_min_molMetal_ = [];
data_table.t_K = [];
data_table.H_T__CO = [];
data_table.H_T__H2O = [];
data_table.H_T__CO2 = [];
data_table.H_T__H2 = [];
data_table.S_T__CO = [];
data_table.S_T__H2O = [];
data_table.S_T__CO2 = [];
data_table.S_T__H2 = [];
data_table.G_T__CO = [];
data_table.G_T__H2O = [];
data_table.G_T__CO2 = [];
data_table.G_T__H2 = [];
data_table.dG_T_ = [];
data_table.dH_T_ = [];
data_table.dS_T_ = [];
data_table.dG_T__1 = [];
data_table.Keq = [];
data_table.H2_vol = [];
data_table.CO2_vol = [];
data_table.CO_vol = [];
data_table.H2O_vol = [];
data_table.H2_flow_in = [];
data_table.CO2_flow_in = [];
data_table.CO_flow_in = [];
data_table.H2O_flow_in = [];
data_table.CO_consumed = [];
data_table.H2_flow_out = [];
data_table.CO2_flow_out = [];
data_table.CO_flow_out = [];
data_table.H2O_flow_out = [];
data_table.Notes = [];
data_table.x1_B = [];
data_table.k_for = [];
data_table.CO_Conv= [];

% because CeO2 is importing as a string for some unknown reason
data_table.CeO2 = str2double(data_table.CeO2);


%% GENERAL CLEANING -------------------------------------------
% -------------------------------------------------------------
% Vectors to store information about the number of deleted data points
data_set_param = [];
data_set_value = [];
data_set_diff = [0];
data_set_param = [data_set_param; 'Initial Data Points'];
data_set_value = [data_set_value; height(data_table)];


% If the support is ZEO then delete that entry
toDelete = data_table.ZEO ~= 0;
data_table(toDelete,:) = [];
data_table.ZEO = [];
    data_set_param = [data_set_param; "ZEO Support"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% If the support is HAP then delete that entry
toDelete = data_table.HAP ~= 0;
data_table(toDelete,:) = [];
data_table.HAP = [];
    data_set_param = [data_set_param; "HAP Support"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% If the support is ACC then delete that entry
toDelete = data_table.ACC ~= 0;
data_table(toDelete,:) = [];
data_table.ACC = [];
    data_set_param = [data_set_param; "ACC Support"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% If the promoter is YSZ then delete that entry
    toDelete = data_table.YSZ ~= 0;
    data_table(toDelete,:) = [];
    data_table.YSZ = [];
        data_set_param = [data_set_param; "YSZ Promoter"];
        data_set_value = [data_set_value; height(data_table)];
        data_set_diff = [data_set_diff; sum(toDelete) ];
        
% Remove any data that had CH4 present
toDelete = data_table.CH4Vol_ ~= 0;
data_table(toDelete,:) = [];
data_table.CH4Vol_ = [];
    data_set_param = [data_set_param; "Contains CH4"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% Remove any data that had O2 present
toDelete = data_table.O2Vol_ ~= 0;
data_table(toDelete,:) = [];
data_table.O2Vol_ = [];
    data_set_param = [data_set_param; "Contains O2"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% Delete rows with Beta > 0.8
toDelete = data_table.beta > 0.8;
data_table(toDelete,:) = [];
    data_set_param = [data_set_param; "Beta > 0.8"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% Delete rows with Beta < 0
toDelete = data_table.beta <= 0;
data_table(toDelete,:) = [];
    data_set_param = [data_set_param; "Beta < 0"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% Delete rows with Beta = NaN
toDelete = ismissing(data_table.beta);
data_table(toDelete,:) = [];
    data_set_param = [data_set_param; "Cannot Calculate Beta"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

% Limit to Low Temp-WGS
toDelete = data_table.T_K < (150+273.15);
data_table(toDelete,:) = [];
    data_set_param = [data_set_param; "T < 150C"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];

toDelete = data_table.T_K > (350+273.15);
data_table(toDelete,:) = [];
    data_set_param = [data_set_param; "T > 350C"];
    data_set_value = [data_set_value; height(data_table)];
    data_set_diff = [data_set_diff; sum(toDelete) ];
  
% Only work with single-support catalysts
    toDelete = data_table.Al2O3 ~= 0 & data_table.Al2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.CeO2 ~= 0 & data_table.CeO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.TiO2 ~= 0 & data_table.TiO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.ZrO2 ~= 0 & data_table.ZrO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.La2O3 ~= 0 & data_table.La2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.MgO ~= 0 & data_table.MgO ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.MnO ~= 0 & data_table.MnO ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Y203 ~= 0 & data_table.Y203 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.SiO2 ~= 0 & data_table.SiO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Fe2O3 ~= 0 & data_table.Fe2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Yb2O3 ~= 0 & data_table.Yb2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.La2O3 ~= 0 & data_table.La2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Tb4O7 ~= 0 & data_table.Tb4O7 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.HfO2 ~= 0 & data_table.HfO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Co3O4 ~= 0 & data_table.Co3O4 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.ThO2 ~= 0 & data_table.ThO2 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Sm2O3 ~= 0 & data_table.Sm2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.Gd2O3 ~= 0 & data_table.Gd2O3 ~= 1;
    data_table(toDelete,:) = [];
    
    toDelete = data_table.CaO ~= 0 & data_table.CaO ~= 1;
    data_table(toDelete,:) = [];
  
        data_set_param = [data_set_param; "Mixed Support"];
        data_set_value = [data_set_value; height(data_table)];
        L = length(data_set_value);
        data_set_diff = [data_set_diff; data_set_value(L-1) - data_set_value(L) ];

%% Evaluate Deleted Data ------------------------------------
% -------------------------------------------------------------

deleted_data = [data_set_param data_set_diff data_set_value];
deleted_data = array2table( deleted_data );


% % Remove any Au / CeO2 data
%     toDelete = data_table.Au ~= 0 & data_table.CeO2 ~= 0;
%     data_table(toDelete,:) = [];
%     disp("Au/CeO2");
%     disp(height(data_table));
%    
% % Remove any Pt / CeO2 data
%     toDelete = data_table.Pt ~= 0 & data_table.CeO2 ~= 0;
%     data_table(toDelete,:) = [];
%     disp("Pt/CeO2");
%     disp(height(data_table));      

%% PRIMARY METAL ----------------------------------------------
% -------------------------------------------------------------

% Use BE of the primary (base) metal

    BE_CO = zeros(height(data_table),1);
    BE_O  = zeros(height(data_table),1);
    BE_C  = zeros(height(data_table),1);
    BE_H  = zeros(height(data_table),1);
    BE_OH = zeros(height(data_table),1);


    Base_Pt = data_table.Pt ~= 0;
        for i=1:length(Base_Pt)
            if Base_Pt(i) == 1
                BE_CO(i) = -1.34;
                BE_O(i)  = -3.18;
                BE_C(i)  = -5.18;
                BE_H(i)  = -2.57;
                BE_OH(i) = -1.65;
            end
        end

    Base_Au = data_table.Au ~= 0;
        for i=1:length(Base_Au)
            if Base_Au(i) == 1
                BE_CO(i) =  0.30;
                BE_O(i)  = -1.94;
                BE_C(i)  = -3.06;
                BE_H(i)  = -1.84;
                BE_OH(i) = -1.06;
            end
        end

    Base_Ru = data_table.Ru ~= 0;
        for i=1:length(Base_Ru)
            if Base_Ru(i) == 1
                BE_CO(i) = -3.47;
                BE_O(i)  = -4.73;
                BE_C(i)  = -6.16;
                BE_H(i)  = -2.75;
                BE_OH(i) = -2.64;
            end
        end

    Base_Rh = data_table.Rh ~= 0;
        for i=1:length(Base_Rh)
            if Base_Rh(i) == 1
                BE_CO(i) = -1.51;
                BE_O(i)  = -4.13;
                BE_C(i)  = -5.67;
                BE_H(i)  = -2.64;
                BE_OH(i) = -2.14;
            end
        end

    Base_Ir = data_table.Ir ~= 0;
        for i=1:length(Base_Ir)
            if Base_Ir(i) == 1
                BE_CO(i) = -1.52;
                BE_O(i)  = -4.14;
                BE_C(i)  = -6.46;
                BE_H(i)  = -2.61;
                BE_OH(i) = -2.12;
            end
        end

    Base_Cu = data_table.Cu ~= 0;
        for i=1:length(Base_Cu)
            if Base_Cu(i) == 1
                BE_CO(i) = -0.33;
                BE_O(i)  = -3.58;
                BE_C(i)  = -3.75;
                BE_H(i)  = -2.20;
                BE_OH(i) = -2.13;
            end
        end

    Base_Pd = data_table.Pd ~= 0;
        for i=1:length(Base_Pd)
            if Base_Pd(i) == 1
                BE_CO(i) = -1.49;
                BE_O(i)  = -3.14;
                BE_C(i)  = -5.62;
                BE_H(i)  = -2.62;
                BE_OH(i) = -1.65;
            end
        end


% Build a single vector with the loading mass fraction

loading_base = zeros(height(data_table),1);

for i=1:height(data_table)
    if data_table.Pt(i) ~= 0
        loading_base(i) = data_table.Pt(i);
    elseif data_table.Au(i) ~= 0
        loading_base(i) = data_table.Au(i);
    elseif data_table.Ru(i) ~= 0
        loading_base(i) = data_table.Ru(i);
    elseif data_table.Rh(i) ~= 0
        loading_base(i) = data_table.Rh(i);
    elseif data_table.Ir(i) ~= 0
        loading_base(i) = data_table.Ir(i);
    elseif data_table.Cu(i) ~= 0
        loading_base(i) = data_table.Cu(i);
    elseif data_table.Pd(i) ~= 0
        loading_base(i) = data_table.Pd(i);
    end
end

loading_base = array2table(loading_base);
BE_C = array2table(BE_C);
BE_CO = array2table(BE_CO);
BE_O  = array2table(BE_O);
BE_H  = array2table(BE_H);
BE_OH = array2table(BE_OH);

data_table = [BE_C, BE_CO, BE_O, BE_H, BE_OH, loading_base, data_table];

% Clean up the data table
data_table.Pt = [];
data_table.Au = [];
data_table.Ru = [];
data_table.Rh = [];
data_table.Ir = [];
data_table.Cu = [];
data_table.Pd = [];

%% PROMOTER ---------------------------------------------------
% -------------------------------------------------------------

% Loading of the Promoter    
    loading_promoter = zeros(height(data_table),1);

    for i=1:height(data_table)
        
        % Group 1
        % Li, Na, K, Rb, Cs
        if data_table.Li(i) ~= 0
            loading_promoter(i) = data_table.Li(i);
        elseif data_table.Na(i) ~= 0
            loading_promoter(i) = data_table.Na(i); 
        elseif data_table.K(i) ~= 0
            loading_promoter(i) = data_table.K(i);
        elseif data_table.Rb(i) ~= 0
            loading_promoter(i) = data_table.Rb(i);
        elseif data_table.Cs(i) ~= 0
            loading_promoter(i) = data_table.Cs(i);
        
        % Group 2
        % Mg, Ca, Sr
        elseif data_table.Mg(i) ~= 0
            loading_promoter(i) = data_table.Mg(i);
        elseif data_table.Ca(i) ~= 0
            loading_promoter(i) = data_table.Ca(i); 
        elseif data_table.Sr(i) ~= 0
            loading_promoter(i) = data_table.Sr(i);
        
        % Group 3
        % Y, La, Ce, Nd, Sm, Gd, Ho, Er, Tm, Yb
        elseif data_table.Y(i) ~= 0
            loading_promoter(i) = data_table.Y(i);
        elseif data_table.La(i) ~= 0
            loading_promoter(i) = data_table.La(i); 
        elseif data_table.Ce(i) ~= 0
            loading_promoter(i) = data_table.Ce(i);
        elseif data_table.Nd(i) ~= 0
            loading_promoter(i) = data_table.Nd(i);
        elseif data_table.Sm(i) ~= 0
            loading_promoter(i) = data_table.Sm(i);
        elseif data_table.Gd(i) ~= 0
            loading_promoter(i) = data_table.Gd(i);
        elseif data_table.Ho(i) ~= 0
            loading_promoter(i) = data_table.Ho(i); 
        elseif data_table.Er(i) ~= 0
            loading_promoter(i) = data_table.Er(i);
        elseif data_table.Tm(i) ~= 0
            loading_promoter(i) = data_table.Tm(i);
        elseif data_table.Yb(i) ~= 0
            loading_promoter(i) = data_table.Yb(i);
            
        % Group 4 to 12 
        % Co, Ni, Re, Ti, Zr, V, Cr, Mn, Fe, Zn    
        elseif data_table.Co(i) ~= 0
            loading_promoter(i) = data_table.Co(i);
        elseif data_table.Ni(i) ~= 0
            loading_promoter(i) = data_table.Ni(i); 
        elseif data_table.Re(i) ~= 0
            loading_promoter(i) = data_table.Re(i);
        elseif data_table.Ti(i) ~= 0
            loading_promoter(i) = data_table.Ti(i);
        elseif data_table.Zr(i) ~= 0
            loading_promoter(i) = data_table.Zr(i);
        elseif data_table.V(i) ~= 0
            loading_promoter(i) = data_table.V(i);
        elseif data_table.Cr(i) ~= 0
            loading_promoter(i) = data_table.Cr(i); 
        elseif data_table.Mn(i) ~= 0
            loading_promoter(i) = data_table.Mn(i);
        elseif data_table.Fe(i) ~= 0
            loading_promoter(i) = data_table.Fe(i);
        elseif data_table.Zn(i) ~= 0
            loading_promoter(i) = data_table.Zn(i);  
        
        end
    end
    
loading_promoter = array2table(loading_promoter);
  

% Describe the Promoters with Materials decriptors
    % First Ionization Energy (eV)
    % Electronegativity
    % Covalent Radius (ang)
    
    FirstIE = zeros( height(data_table), 1) ;
    Electronegativity = zeros( height(data_table), 1) ;
    CovalentRadius = zeros( height(data_table), 1) ;
    ChargeLow = zeros( height(data_table), 1) ;
    ChargeHigh = zeros( height(data_table), 1) ;
    Z_IonicRad = zeros( height(data_table), 1) ;
    Redox_Prom = zeros( height(data_table), 1) ;
    MW_Prom = zeros( height(data_table), 1) ;
    
    % Group 1 ------
    % Li, Na, K, Rb, Cs
    
    toChange = data_table.Li ~= 0;
        FirstIE(toChange,:) = 5.39;
        Electronegativity(toChange,:) = 0.98;
        CovalentRadius(toChange,:) = 1.28;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 1;
        Z_IonicRad (toChange,:) = 0.1316;
        Redox_Prom (toChange,:) = -3.040;
        MW_Prom (toChange,:) = 6.94;
        data_table.Li = [];
        
    toChange = data_table.Na ~= 0;
        FirstIE(toChange,:) = 5.14;
        Electronegativity(toChange,:) = 0.93;
        CovalentRadius(toChange,:) = 1.66;
        ChargeLow (toChange,:) = -1;
        ChargeHigh (toChange,:) = 1;
        Z_IonicRad (toChange,:) = 0.0980;
        Redox_Prom (toChange,:) = -2.710;
        MW_Prom (toChange,:) = 22.99;
        data_table.Na = [];
        
    toChange = data_table.K ~= 0;
        FirstIE(toChange,:) = 4.34;
        Electronegativity(toChange,:) = 0.82;
        CovalentRadius(toChange,:) = 2.03;
        ChargeLow (toChange,:) = -1;
        ChargeHigh (toChange,:) = 1;
        Z_IonicRad (toChange,:) = 0.0725;
        Redox_Prom (toChange,:) = -2.931;
        MW_Prom (toChange,:) = 39.10;
        data_table.K = [];
        
    toChange = data_table.Rb ~= 0;
        FirstIE(toChange,:) =4.18 ;
        Electronegativity(toChange,:) = 0.82;
        CovalentRadius(toChange,:) = 2.20;
        ChargeLow (toChange,:) = -1;
        ChargeHigh (toChange,:) = 1;
        Z_IonicRad (toChange,:) = 0.0658;
        Redox_Prom (toChange,:) = -2.980;
        MW_Prom (toChange,:) = 85.47;
        data_table.Rb = [];
        
    toChange = data_table.Cs ~= 0;
        FirstIE(toChange,:) = 3.89;
        Electronegativity(toChange,:) = 0.79;
        CovalentRadius(toChange,:) = 2.44;
        ChargeLow (toChange,:) = -1;
        ChargeHigh (toChange,:) = 1;
        Z_IonicRad (toChange,:) = 0.0599;
        Redox_Prom (toChange,:) = -3.026;
        MW_Prom (toChange,:) = 132.90;
        data_table.Cs = [];

        
    % Group 2 ------
    % Mg, Ca, Sr
    
    toChange = data_table.Mg ~= 0;
        FirstIE(toChange,:) = 7.65;
        Electronegativity(toChange,:) = 1.31;
        CovalentRadius(toChange,:) = 1.41;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 2;
        Z_IonicRad (toChange,:) = 0.2778;
        Redox_Prom (toChange,:) = -2.372;
        MW_Prom (toChange,:) = 24.305;
        data_table.Mg = [];
        
    toChange = data_table.Ca ~= 0;
        FirstIE(toChange,:) = 6.11;
        Electronegativity(toChange,:) = 1.00;
        CovalentRadius(toChange,:) = 1.76;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 2;
        Z_IonicRad (toChange,:) = 0.2000;
        Redox_Prom (toChange,:) = -2.868;
        MW_Prom (toChange,:) = 40.078;
        data_table.Ca = [];
        
    toChange = data_table.Sr ~= 0;
        FirstIE(toChange,:) = 5.70;
        Electronegativity(toChange,:) = 0.95;
        CovalentRadius(toChange,:) = 1.95;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 2;
        Z_IonicRad (toChange,:) = 0.1695;
        Redox_Prom (toChange,:) = -2.899;
        MW_Prom (toChange,:) = 87.62;
        data_table.Sr = [];
        
        
    % Group 3 ------
    % Y, La, Ce, Nd, Sm, Gd, Ho, Er, Tm, Yb
    
     toChange = data_table.Y ~= 0;
        FirstIE(toChange,:) = 6.22;
        Electronegativity(toChange,:) = 1.22;
        CovalentRadius(toChange,:) = 1.76;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3333;
        Redox_Prom (toChange,:) = -2.372;
        MW_Prom (toChange,:) = 88.91;
        data_table.Y = [];
    
     toChange = data_table.La ~= 0;
        FirstIE(toChange,:) = 5.58;
        Electronegativity(toChange,:) = 1.10;
        CovalentRadius(toChange,:) = 2.07;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.2907;
        Redox_Prom (toChange,:) = -2.379;
        MW_Prom (toChange,:) = 138.91;
        data_table.La = [];
        
    toChange = data_table.Ce ~= 0;
        FirstIE(toChange,:) = 5.54;
        Electronegativity(toChange,:) = 1.12;
        CovalentRadius(toChange,:) = 2.04;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.4598;
        Redox_Prom (toChange,:) = -2.336;
        MW_Prom (toChange,:) = 140.12;
        data_table.Ce = [];
        
    toChange = data_table.Nd ~= 0;
        FirstIE(toChange,:) = 5.53;
        Electronegativity(toChange,:) = 1.14;
        CovalentRadius(toChange,:) = 2.01;
        ChargeLow (toChange,:) = 2;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.4069;
        Redox_Prom (toChange,:) = -2.323;
        MW_Prom (toChange,:) = 144.24;
        data_table.Nd = [];
        
    toChange = data_table.Sm ~= 0;
        FirstIE(toChange,:) = 5.64;
        Electronegativity(toChange,:) = 1.17;
        CovalentRadius(toChange,:) = 1.98;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.4175;
        Redox_Prom (toChange,:) = -2.304;
        MW_Prom (toChange,:) = 150.36;
        data_table.Sm = [];
        
    toChange = data_table.Gd ~= 0;
        FirstIE(toChange,:) = 6.15;
        Electronegativity(toChange,:) = 1.20;
        CovalentRadius(toChange,:) = 1.96;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3209;
        Redox_Prom (toChange,:) = -2.279;
        MW_Prom (toChange,:) = 157.25;
        data_table.Gd = [];
    
     toChange = data_table.Ho ~= 0;
        FirstIE(toChange,:) = 6.02;
        Electronegativity(toChange,:) = 1.23;
        CovalentRadius(toChange,:) = 1.92;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3330;
        Redox_Prom (toChange,:) = -2.330;
        MW_Prom (toChange,:) = 164.93;
        data_table.Ho = [];
        
    toChange = data_table.Er ~= 0;
        FirstIE(toChange,:) = 6.11;
        Electronegativity(toChange,:) = 1.24;
        CovalentRadius(toChange,:) = 1.89;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3371;
        Redox_Prom (toChange,:) = -2.331;
        MW_Prom (toChange,:) = 167.26;
        data_table.Er = [];
        
    toChange = data_table.Tm ~= 0;
        FirstIE(toChange,:) = 6.18;
        Electronegativity(toChange,:) = 1.25;
        CovalentRadius(toChange,:) = 1.90;
        ChargeLow (toChange,:) = 2;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3409;
        Redox_Prom (toChange,:) = -2.319;
        MW_Prom (toChange,:) = 168.93;
        data_table.Tm = [];
        
    toChange = data_table.Yb ~= 0;
        FirstIE(toChange,:) = 5.43;
        Electronegativity(toChange,:) = 1.27;
        CovalentRadius(toChange,:) = 1.87;
        ChargeLow (toChange,:) = 1;
        ChargeHigh (toChange,:) = 3;
        Z_IonicRad (toChange,:) = 0.3456;
        Redox_Prom (toChange,:) = -2.190;
        MW_Prom (toChange,:) = 174.97;
        data_table.Yb = [];
    
    % Group 4 to 12 ------
    % Co, Ni, Re, Ti, Zr, V, Cr, Mn, Fe, Zn
    
     toChange = data_table.Co ~= 0;
        FirstIE(toChange,:) = 7.88;
        Electronegativity(toChange,:) = 1.88;
        CovalentRadius(toChange,:) = 1.18;
        ChargeLow (toChange,:) = -3;
        ChargeHigh (toChange,:) = 5;
        Z_IonicRad (toChange,:) = 0.9174;
        Redox_Prom (toChange,:) = -0.28;
        MW_Prom (toChange,:) = 58.93;
        data_table.Co = [];
        
    toChange = data_table.Ni ~= 0;
        FirstIE(toChange,:) = 7.64;
        Electronegativity(toChange,:) = 1.91;
        CovalentRadius(toChange,:) = 1.17;
        ChargeLow (toChange,:) = -4;
        ChargeHigh (toChange,:) = 2;
        Z_IonicRad (toChange,:) = 0.3571;
        Redox_Prom (toChange,:) = -0.26;
        MW_Prom (toChange,:) = 58.69;
        data_table.Ni = [];
        
    toChange = data_table.Re ~= 0;
        FirstIE(toChange,:) = 7.83;
        Electronegativity(toChange,:) = 1.90;
        CovalentRadius(toChange,:) = 1.41;
        ChargeLow (toChange,:) = -3;
        ChargeHigh (toChange,:) = 7;
        Z_IonicRad (toChange,:) = 1.3208;
        Redox_Prom (toChange,:) = 0.30;
        MW_Prom (toChange,:) = 186.21;
        data_table.Re = [];
        
    toChange = data_table.Ti ~= 0;
        FirstIE(toChange,:) = 13.58;
        Electronegativity(toChange,:) = 1.54;
        CovalentRadius(toChange,:) = 1.48;
        ChargeLow (toChange,:) = -2;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.6612;
        Redox_Prom (toChange,:) = -1.370;
        MW_Prom (toChange,:) = 47.87;
        data_table.Ti = [];
        
    toChange = data_table.Zr ~= 0;
        FirstIE(toChange,:) = 6.63;
        Electronegativity(toChange,:) = 1.33;
        CovalentRadius(toChange,:) = 1.64;
        ChargeLow (toChange,:) = -2;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.5556;
        Redox_Prom (toChange,:) = -1.450;
        MW_Prom (toChange,:) = 91.22;
        data_table.Zr = [];
        
    toChange = data_table.V ~= 0;
        FirstIE(toChange,:) = 6.75;
        Electronegativity(toChange,:) = 1.63;
        CovalentRadius(toChange,:) = 1.44;
        ChargeLow (toChange,:) = -2;
        ChargeHigh (toChange,:) = 4;
        Z_IonicRad (toChange,:) = 0.9259;
        Redox_Prom (toChange,:) = -1.175;
        MW_Prom (toChange,:) = 50.94;
        data_table.V = [];
        
    toChange = data_table.Cr ~= 0;
        FirstIE(toChange,:) = 6.77;
        Electronegativity(toChange,:) = 1.66;
        CovalentRadius(toChange,:) = 1.30;
        ChargeLow (toChange,:) = -4;
        ChargeHigh (toChange,:) = 6;
        Z_IonicRad (toChange,:) = 1.3636;
        Redox_Prom (toChange,:) = -0.744;
        MW_Prom (toChange,:) = 52.00;
        data_table.Cr = [];
        
    toChange = data_table.Mn ~= 0;
        FirstIE(toChange,:) = 7.43;
        Electronegativity(toChange,:) = 1.55;
        CovalentRadius(toChange,:) = 1.29;
        ChargeLow (toChange,:) = -3;
        ChargeHigh (toChange,:) = 7;
        Z_IonicRad (toChange,:) = 1.5217;
        Redox_Prom (toChange,:) = -1.185;
        MW_Prom (toChange,:) = 54.95;
        data_table.Mn = [];
        
    toChange = data_table.Fe ~= 0;
        FirstIE(toChange,:) = 7.90;
        Electronegativity(toChange,:) = 1.83;
        CovalentRadius(toChange,:) = 1.24;
        ChargeLow (toChange,:) = -4;
        ChargeHigh (toChange,:) = 7;
        Z_IonicRad (toChange,:) = 1.0853;
        Redox_Prom (toChange,:) = -0.037;
        MW_Prom (toChange,:) = 55.85;
        data_table.Fe = [];
        
    toChange = data_table.Zn ~= 0;
        FirstIE(toChange,:) = 9.39;
        Electronegativity(toChange,:) = 1.65;
        CovalentRadius(toChange,:) = 1.20;
        ChargeLow (toChange,:) = -2;
        ChargeHigh (toChange,:) = 2;
        Z_IonicRad (toChange,:) = 0.2703;
        Redox_Prom (toChange,:) = -0.762;
        MW_Prom (toChange,:) = 65.38;
        data_table.Zn = [];

        
FirstIE = array2table(FirstIE);
Electronegativity = array2table(Electronegativity);
CovalentRadius = array2table(CovalentRadius);
ChargeLow = array2table(ChargeLow);
ChargeHigh = array2table(ChargeHigh);
Z_IonicRad = array2table(Z_IonicRad);
Redox_Prom = array2table(Redox_Prom);
MW_Prom = array2table(MW_Prom);

insert =6; 
data_table = [data_table(:,1:insert), MW_Prom, FirstIE, CovalentRadius, ChargeLow, ChargeHigh, Z_IonicRad, Electronegativity, Redox_Prom, loading_promoter, data_table(:,(insert+1):end)];

%% SUPPORTS ---------------------------------------------------
% -------------------------------------------------------------
    
    
% Describe the Supports with Materials decriptors
    % Redox properties
    % First ionization energy of the metal
    % Electronegativity
    % Highest Oxidation State
    % Lowest Oxidation State
    % Molecular Weight of the metal
    % Molecular weight of the whole compound

    toChange = data_table.Al2O3 ~= 0;
        Redox(toChange,:) = -1.662;
        FirstIE_supp(toChange,:) = 577.5; % kJ/mol
        Electroneg_supp(toChange,:) = 1.61;
        OxidState_high_supp(toChange,:) = 3;
        OxidState_low_supp(toChange,:) = -2;
        MW_supp(toChange,:) = 26.982;
        MW_comp(toChange,:) = 2* 29.982 + 3* 15.999;


    toChange = data_table.CeO2 ~= 0;
         Redox(toChange,:) = -2.336;
         FirstIE_supp(toChange,:) = 534.4;
         Electroneg_supp(toChange,:) = 1.12;
         OxidState_high_supp(toChange,:) = 4;
         OxidState_low_supp(toChange,:) = 1;
         MW_supp(toChange,:) = 140.12;
         MW_comp(toChange,:) = 140.12 + 2 * 15.999;

    toChange = data_table.TiO2 ~= 0;
        Redox(toChange,:) = -1.63;
        FirstIE_supp(toChange,:) = 658.8;
        Electroneg_supp(toChange,:) = 1.54;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = -2;
        MW_supp(toChange,:) = 47.867;
        MW_comp(toChange,:) = 47.867 + 2 * 15.999; 


    toChange = data_table.ZrO2 ~= 0;
        Redox(toChange,:) = -1.45;
        FirstIE_supp(toChange,:) = 640.1;
        Electroneg_supp(toChange,:) = 1.33;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = -2;
        MW_supp(toChange,:) = 91.224;
        MW_comp(toChange,:) = 91.224 + 2 * 15.999;


    toChange = data_table.La2O3 ~= 0;
        Redox(toChange,:) = -2.379;
        FirstIE_supp(toChange,:) = 538.1;
        Electroneg_supp(toChange,:) = 1.10;
        OxidState_high_supp(toChange,:) = 3;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 138.91;
        MW_comp(toChange,:) = 2* 138.91 + 3 * 15.999;
        
    toChange = data_table.MgO ~= 0;
        Redox(toChange,:) = -2.7;
        FirstIE_supp(toChange,:) = 737.7;
        Electroneg_supp(toChange,:) = 1.31;
        OxidState_high_supp(toChange,:) = 2;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 24.305;
        MW_comp(toChange,:) = 24.305 + 1 * 15.999;
        
    toChange = data_table.MnO ~= 0;
        Redox(toChange,:) = -1.185;
        FirstIE_supp(toChange,:) = 717.3;
        Electroneg_supp(toChange,:) = 1.55;
        OxidState_high_supp(toChange,:) = 7;
        OxidState_low_supp(toChange,:) = -3;
        MW_supp(toChange,:) = 54.938;
        MW_comp(toChange,:) = 54.938 + 1 * 15.999;
        
    toChange = data_table.Y203 ~= 0;
        Redox(toChange,:) = -2.372;
        FirstIE_supp(toChange,:) = 600;
        Electroneg_supp(toChange,:) = 1.22;
        OxidState_high_supp(toChange,:) = 3;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 88.906;
        MW_comp(toChange,:) = 2 * 88.906 + 3 * 15.999;
        
    toChange = data_table.SiO2 ~= 0;
        Redox(toChange,:) = 0.857;
        FirstIE_supp(toChange,:) = 786.5;
        Electroneg_supp(toChange,:) = 1.9;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = -4;
        MW_supp(toChange,:) = 28.085;
        MW_comp(toChange,:) = 28.085 + 2 * 15.999;
        
    toChange = data_table.Fe2O3 ~= 0;
        Redox(toChange,:) = -0.037;
        FirstIE_supp(toChange,:) = 762.5;
        Electroneg_supp(toChange,:) = 1.83;
        OxidState_high_supp(toChange,:) = 7;
        OxidState_low_supp(toChange,:) = -4;
        MW_supp(toChange,:) = 55.845;
        MW_comp(toChange,:) = 2 * 55.845 + 3 * 15.999;
        
    toChange = data_table.Yb2O3 ~= 0;
        Redox(toChange,:) = -2.19;
        FirstIE_supp(toChange,:) = 603.4;
        Electroneg_supp(toChange,:) = 1.1;
        OxidState_high_supp(toChange,:) = 3;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 88.906;
        MW_comp(toChange,:) = 2* 88.906 + 3 * 15.999;
        
       
    toChange = data_table.Tb4O7 ~= 0;
        Redox(toChange,:) = -2.28;
        FirstIE_supp(toChange,:) = 565.8;
        Electroneg_supp(toChange,:) = 1.2;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 158.93;
        MW_comp(toChange,:) = 4* 158.93 + 7 * 15.999;
    
    
    toChange = data_table.HfO2 ~= 0;
        Redox(toChange,:) = -1.55;
        FirstIE_supp(toChange,:) = 658.5;
        Electroneg_supp(toChange,:) = 1.3;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = -2;
        MW_supp(toChange,:) = 178.49;
        MW_comp(toChange,:) = 178.49 + 2 * 15.999;
    
    
    toChange = data_table.Co3O4 ~= 0;
        Redox(toChange,:) = -0.28;
        FirstIE_supp(toChange,:) = 760.4;
        Electroneg_supp(toChange,:) = 1.88;
        OxidState_high_supp(toChange,:) = 5;
        OxidState_low_supp(toChange,:) = -3;
        MW_supp(toChange,:) = 58.693;
        MW_comp(toChange,:) = 3 * 58.693 + 4 * 15.999;
    
    
    toChange = data_table.ThO2 ~= 0;
        Redox(toChange,:) = -1.899;
        FirstIE_supp(toChange,:) = 587;
        Electroneg_supp(toChange,:) = 1.3;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 232.04;
        MW_comp(toChange,:) = 232.04 + 2 * 15.999;
    
    
    toChange = data_table.Sm2O3 ~= 0;
        Redox(toChange,:) = -2.304;
        FirstIE_supp(toChange,:) = 544.5;
        Electroneg_supp(toChange,:) = 1.17;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 150.36;
        MW_comp(toChange,:) = 2* 150.36 + 3 * 15.999;
    
    
    toChange = data_table.Gd2O3 ~= 0;
        Redox(toChange,:) = -2.297;
        FirstIE_supp(toChange,:) = 593.4;
        Electroneg_supp(toChange,:) = 1.2;
        OxidState_high_supp(toChange,:) = 3;
        OxidState_low_supp(toChange,:) = 1;
        MW_supp(toChange,:) = 157.25;
        MW_comp(toChange,:) = 2* 157.25 + 3 * 15.999;
    
    
    toChange = data_table.CaO ~= 0;
        Redox(toChange,:) = -3.8;
        FirstIE_supp(toChange,:) = 640.1;
        Electroneg_supp(toChange,:) = 1.33;
        OxidState_high_supp(toChange,:) = 4;
        OxidState_low_supp(toChange,:) = -2;
        MW_supp(toChange,:) = 40.078;
        MW_comp(toChange,:) = 40.078 + 1 * 15.999;
        
        

        
Redox = array2table(Redox);
FirstIE_supp = FirstIE_supp./1000;
FirstIE_supp = array2table(FirstIE_supp);
Electroneg_supp = array2table(Electroneg_supp);
OxidState_high_supp = array2table(OxidState_high_supp);
OxidState_low_supp = array2table(OxidState_low_supp);
MW_supp = array2table(MW_supp);
MW_comp = array2table(MW_comp);

insert = 16;
data_table = [data_table(:, 1:insert),MW_comp, MW_supp, FirstIE_supp, Redox,...
    OxidState_low_supp, OxidState_high_supp, Electroneg_supp,  data_table(:, (insert+1):end)];

data_table.Al2O3 = [];
data_table.CeO2 = [];
data_table.TiO2 = [];
data_table.ZrO2 = [];
data_table.La2O3 = [];
data_table.MgO = [];
data_table.MnO = [];
data_table.Y203 = [];
data_table.SiO2 = [];
data_table.Fe2O3 = [];
data_table.Yb2O3 = [];
data_table.Tb4O7 = [];
data_table.HfO2 = [];
data_table.Co3O4 = [];
data_table.ThO2 = [];
data_table.Sm2O3 = [];
data_table.Gd2O3 = [];
data_table.CaO = [];

%% SYNTHESIS --------------------------------------------------
% -------------------------------------------------------------
% Clean up synthesis methods

if any(data_table.IWI) == 0
    data_table.IWI = [];
end

if any(data_table.WI) == 0
    data_table.WI = [];
end

if any(data_table.CI) == 0
    data_table.CI = [];
end

if any(data_table.SI) == 0
    data_table.SI = [];
end

if any(data_table.SGP) == 0
    data_table.SGP = [];
end

if any(data_table.CP) == 0
    data_table.CP = [];
end
if any(data_table.HDP) == 0
    data_table.HDP = [];
end

if any(data_table.UGC) == 0
    data_table.UGC = [];
end

if any(data_table.SCT) == 0
    data_table.SCT = [];
end

if any(data_table.FSP) == 0
    data_table.FSP = [];
end

if any(data_table.ME) == 0
    data_table.ME = [];
end

if any(data_table.DP) == 0
    data_table.DP = [];
end


%% -------------------------------------------------------------
% Calculate rate/(1-B)

rate_for = zeros(length(data_table.Rate_molsCO_min_molMetal_),1);
rate_for_adj = zeros(length(data_table.Rate_molsCO_min_molMetal_),1);
Ea = 70; % kJ/mol
Rgas = 8.314 * 10^-3; % kJ/mol/K
T_ref = 250+273.15; % K


for i = 1: length(data_table.Rate_molsCO_min_molMetal_)
    rate_for(i) = data_table.Rate_molsCO_min_molMetal_(i) / ( 1-data_table.beta(i) ) ;
    T = data_table.T_K(i);
    CorrectionFactor = exp(Ea/Rgas * (1/T - 1/T_ref) );
    rate_for_adj(i) = rate_for(i) * CorrectionFactor;
end

rate_for = array2table(rate_for);
rate_for_adj = array2table(rate_for_adj);

data_table = [data_table, rate_for_adj];

data_table.Rate_molsCO_min_molMetal_ = [];

%% Remove data with an attribute that has limited data

X = 20;
    
% Synthesis Method
    if nnz(data_table.IWI) < X
        toDelete = data_table.IWI ~= 0;
        data_table(toDelete,:) = [];
        data_table.IWI = [];
        disp("Removed synthesis, IWI")
    end
    
    if nnz(data_table.WI) < X
        toDelete = data_table.WI ~= 0;
        data_table(toDelete,:) = [];
        data_table.WI = [];
        disp("Removed synthesis, WI")
    end
    
    if nnz(data_table.CI) < X
        toDelete = data_table.CI ~= 0;
        data_table(toDelete,:) = [];
        data_table.CI = [];
        disp("Removed synthesis, CI")
    end
    
    if nnz(data_table.SI) < X
        toDelete = data_table.SI ~= 0;
        data_table(toDelete,:) = [];
        data_table.SI = [];
        disp("Removed synthesis, SI")
    end
    
    
    if nnz(data_table.SGP) < X
        toDelete = data_table.SGP ~= 0;
        data_table(toDelete,:) = [];
        data_table.SGP = [];
        disp("Removed synthesis, SGP")
    end
    
    if nnz(data_table.CP) < X
        toDelete = data_table.CP ~= 0;
        data_table(toDelete,:) = [];
        data_table.CP = [];
        disp("Removed synthesis, CP")
    end
    
    
    if nnz(data_table.HDP) < X
        toDelete = data_table.HDP ~= 0;
        data_table(toDelete,:) = [];
        data_table.HDP = [];
        disp("Removed synthesis, HDP")
    end
    
    
    if nnz(data_table.DP) < X
        toDelete = data_table.DP ~= 0;
        data_table(toDelete,:) = [];
        data_table.DP = [];
        disp("Removed synthesis, DP")
    end
    
        data_set_param = [data_set_param; "Synthesis"];
        data_set_value = [data_set_value; height(data_table)];
        L = length(data_set_value);
        data_set_diff = [data_set_diff; data_set_value(L-1) - data_set_value(L) ];
    
%% Normalize the output
% this will scale the output s.t. it is well distributed throughout the
% domain. Then, outliers are removed. 

data_table.rate_for_adj_norm = log(data_table.rate_for_adj);

% Remove Outliers
toDelete = data_table.rate_for_adj_norm > 10;
data_table(toDelete,:) = [];
toDelete = data_table.rate_for_adj_norm < -10;
data_table(toDelete,:) = [];

  
writetable(data_table,'final_withPtAuCeO2.txt')


%% Temperature Distribution ------------------------------------
% -------------------------------------------------------------
% Bar Plot
T_C = data_table.T_K - 273;

x = [];
y = [];
x_lab = [];
inc = 20;
for i = (150 + inc):inc:350
    x = [x;i];
    x_lab = [x_lab;strcat( num2str(i-inc), "-", num2str(i))];
    y = [y; sum(T_C > (i-inc) & T_C <= i)];
end


figure('DefaultAxesFontSize',18)
bar(x,y)
% title('Temperature Distribution');
xlabel('Temperature (C)');
xticklabels(x_lab);
xtickangle(45);
xlim([150,370])
ylabel('Frequency');

% Scatter/Line plot

x = [];
y = [];
inc = 10;
for i = 150:inc:350
    x = [x;i];
    y = [y; sum(T_C > (0) & T_C <= i)];
end
figure('DefaultAxesFontSize',18)
plot(x,y,...
    'LineWidth',3)
% title('Temperature Distribution');
xlabel('Temperature (C)');
xtickangle(45);
xlim([150,350])
ylabel('Cumulative Frequency');

%% Beta Distribution ------------------------------------------
% -------------------------------------------------------------
% Bar Plot

x = [];
y = [];
x_lab = [];
inc = 0.08;
for i = 0.08:inc:0.8
    x = [x;i];
    x_lab = [x_lab;strcat( num2str(i-inc), "-", num2str(i))];
    y = [y; sum(data_table.beta > (i - inc) & data_table.beta <= i )];
end

figure('DefaultAxesFontSize',16)
bar(y)
ylim([0 200]);
% title('Beta Distribution');
xticklabels(x_lab);
xtickangle(45);
xlabel('Beta Value');
ylabel('Frequency');

% Scatter/Line plot

x = [];
y = [];
for i = 0:0.01:0.8
    x = [x;i];
    y = [y; sum(data_table.beta > (0) & data_table.beta <= i )];
end

figure('DefaultAxesFontSize',16)
plot(x,y,...
    'LineWidth',3)
% title('Beta Distribution');
xtickangle(45);
xlabel('Beta Value');
ylabel('Cumulative Frequency');

data_table.beta = [];

%% Output Data Distribution -----------------------------------
% -------------------------------------------------------------

% Non-Normalized histogram
x = [];
y = [];
x_lab = [];
inc = 20;
for i = 0+inc:inc:2000
    x = [x;i];
    x_lab = [x_lab;strcat( num2str(i-inc), "-", num2str(i))];
    y = [y; sum(data_table.rate_for_adj > (i - inc) & data_table.rate_for_adj <= i )];
end

figure('DefaultAxesFontSize',18)
bar(x,y)
ylim([0 20]);
xlim([0 2000]);
% title('Forward Rate Distribution');
% xticklabels(x);
xtickangle(45);
xlabel('Forward Rate Value');
ylabel('Frequency');

% 
% % Normalized Data Distribution
% x = 1:1:length(data_table.rate_for_adj_norm);
% y = data_table.rate_for_adj_norm;
% plot(x,y);


% Normalized Histogram
x = [];
y = [];
x_lab = [];
inc = 0.5;
for i = -8+inc:inc:8
    x = [x;i];
    x_lab = [x_lab;strcat( num2str(i-inc), " to ", num2str(i))];
    y = [y; sum(data_table.rate_for_adj_norm > (i - inc) & data_table.rate_for_adj_norm <= i )];
end

figure('DefaultAxesFontSize',18)
bar(x,y)
% title('Normalized Forward Rate Distribution');
% xticklabels(x);
xtickangle(45);
xlim([-8 8]);
xlabel('Normalized Forward Rate Value');
ylabel('Frequency');



import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

uFile = 'Thermal_Calcs_DF.sav'
dfU = pd.read_pickle(uFile)

pFile = 'Product_DF_v2.sav'
dfP = pd.read_pickle(pFile)

#frames = dfP.columns.values.tolist()
#frames[0] = 'Generic product'
states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
       'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
       'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
       'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
       'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

wFile = 'Weather_Cities_DF.sav'
dfW = pd.read_pickle(wFile)
#st.dataframe(dfW)

def run_calc(dfW, dfP):

    file1 = 'gas_model3.sav'
    gas_model = pickle.load(open(file1, 'rb'))

    file2 = 'elec_model3.sav'
    elec_model = pickle.load(open(file2, 'rb'))

    file3 = 'elec_model_all_building3.sav'
    elec_blg_model = pickle.load(open(file3, 'rb'))

    file4 = 'gas_model_all_building3.sav'
    gas_blg_model = pickle.load(open(file4, 'rb'))

    dFile = 'Deploy_Weather_DF2.sav'
    df = pd.read_pickle(dFile)

    cFile = 'Energy_Rates_DF.sav'
    dfC = pd.read_pickle(cFile)

    bFile = 'Building_DF.sav'
    dfB = pd.read_pickle(bFile)

    wid = int(dfW['Weather ID'].loc[(dfW['City']) == st.session_state.city])
    #file3 = 'energy_model.sav'
    area = float(st.session_state.area)*0.09
    blg = st.session_state.blg
    flr = int(st.session_state.flr_slider)
    wwr = st.session_state.wwr_slider
    u_val1 = st.session_state.u1_slider*5.7
    yr = st.session_state.yr_slider
    shgc1 = st.session_state.s1_slider
    loc = st.session_state.city
    if st.session_state.win2:
        u_val2 = st.session_state.u2_slider*5.7
        shgc2 = st.session_state.s2_slider
    #x = [area, blg, flr, wwr, u_val1, shgc, loc]
    #---floor to floor height [m]---
    ftfh = 4
    side = (area/flr)**(1/2)
    height = flr*4
    wall_area = 4*height*side
    volume = height*side**2
    win_area = wwr*wall_area
    south_win_area = 0.3*win_area
    #st.table(wid)
    df = df.loc[(df['Weather ID']==wid)]
    dfB = dfB.loc[(dfB['Building Name']==st.session_state.blg)]
    bid = float(dfB['BID'])

    df['Total Irradience'] = df['Direct Irradience'] + df['Diffuse Irradience']

    # X2 = df[['BID','Weather ID','Week','Floor Area', 'Above ground wall area',
    #    'Total Window Area',
    #    'Window U-factor', 'Solar Heat Gain Coefficient',
    #    'Building Volume',
    #    'South window area',
    #    'Cooling Degree Days',
    #    'Heating Degree Days', 'Direct Irradience']]

    df['BID'] = bid
    df['Floor Area'] = area
    df['Window to Wall Ratio'] = wwr
    df['Above ground wall area'] = wall_area
    df['Total Window Area'] = win_area
    df['Building Volume'] = volume
    df['Skylight area'] = float(dfB['Skylight area'])
    df['Window U-factor'] = u_val1 
    df['Solar Heat Gain Coefficient'] = shgc1
    df['Plug and process'] = float(dfB['Plug and process'])
    df['Lighting'] = float(dfB['Lighting'])
    df['People'] = float(dfB['People'])
    df['South window area'] = south_win_area
    df['Gross roof area'] = area/flr


    df['Floor Area'] = df['Floor Area']/1000
    df['Total Window Area'] = df['Total Window Area']/1000
    df['Above ground wall area'] = df['Above ground wall area']/1000
    df['Building Volume'] = df['Building Volume']/1000
    df['South window area'] = df['South window area']/1000
    df['Total Irradience'] = df['Total Irradience']/1000
    df['Climate Zone Letter'] = 1
    df['Wall U-factor'] = 0.38
    df['Roof U-factor'] = 0.2
    df['South wall area'] = wall_area/4

    x = df[['BID','Climate zone number', 'Month',
       'Floor Area', 'Window to Wall Ratio', 'Above ground wall area',
       'Total Window Area', 'Building Volume', 'Skylight area',
       'Window U-factor', 'Solar Heat Gain Coefficient',
       'Plug and process', 'Lighting', 'People',
       'South window area', 'Avg Noon RH',
       'Cooling Degree Days',
       'Heating Degree Days', 'Total Irradience']]

    x3 = df[['BID',
       'Climate zone number', 'Climate zone letter', 'Month', 'Week',
       'Floor Area', 'Above ground wall area',
       'Total Window Area', 'Window to Wall Ratio',
       'Window U-factor', 'Solar Heat Gain Coefficient',
       'Building Volume', 'South wall area','South window area',
       'Gross roof area','Skylight area', 'Lighting', 'People',
       'Plug and process', 'Wall U-factor',
       'Roof U-factor', 'Cooling Degree Days',
       'Heating Degree Days', 'Direct Irradience',
       'Diffuse Irradience', 'Wind Speed', 'Avg Noon RH']]
    #st.dataframe(x2)

    y_gas = gas_model.predict(x)
    y_elec = elec_model.predict(x)

    y_blg_gas = gas_blg_model.predict(x3)
    y_blg_elec = elec_blg_model.predict(x3)
    #st.text(str(y_gas))
    #st.text(str(y_elec))
    #---Window energy density [MJ/m²] over lifetime---
    elec_total = y_elec.sum()-10*52
    gas_total = y_gas.sum()-100*52
    elec_blg_total = y_blg_elec.sum()
    gas_blg_total = y_blg_gas.sum()

    if st.session_state.win2:
        x2 = x.copy()
        x4 = x3.copy()
        x2['Window U-factor'] = u_val2
        x2['Solar Heat Gain Coefficient'] = shgc2
        x4['Window U-factor'] = u_val2
        x4['Solar Heat Gain Coefficient'] = shgc2
        y_gas2 = gas_model.predict(x2)
        y_elec2 = elec_model.predict(x2)

        y_blg_gas2 = gas_blg_model.predict(x4)
        y_blg_elec2 = elec_blg_model.predict(x4)

        elec_total2 = y_elec2.sum()-10*52
        gas_total2 = y_gas2.sum()-100*52

        elec_blg_total2 = y_blg_elec2.sum()
        gas_blg_total2 = y_blg_gas2.sum()
        #---Carbon calculations---
        #---Electricity operational carbon [kgCO2/m²]---
        elec_carbon2 = yr*float(dfC['kgCO2/MWh'].loc[(dfC['State']==st.session_state.states)])*elec_total2/3600
        #---Natural gas operational carbon [kgCO2/m²]---
        gas_carbon2 = yr*54.15*gas_total2/1000
        op_carbon2 = elec_carbon2 + gas_carbon2
        op_annual2 = op_carbon2/yr
    else:
        elec_carbon2 = 0
        gas_carbon2 = 0
        op_carbon2 = 0    
    #---Carbon calculations---
    #---Electricity operational carbon [kgCO2/m²]---
    elec_carbon1 = yr*float(dfC['kgCO2/MWh'].loc[(dfC['State']==st.session_state.states)])*elec_total/3600
    #---Natural gas operational carbon [kgCO2/m²]---
    gas_carbon1 = yr*54.15*gas_total/1000
    op_carbon1 = elec_carbon1 + gas_carbon1
    op_annual1 = op_carbon1/yr

    #-----calculate energy and cost----
    #---Energy consumption [kWh/m²]---
    annual_kwh1 = (elec_total + gas_total)*277.8/1000
    annual_kwh2 = (elec_total2 + gas_total2)*277.8/1000
    annual_kwh_diff1 = annual_kwh1 - annual_kwh2
    annual_kwh_diff2 = (elec_blg_total - elec_blg_total2 + gas_blg_total - gas_blg_total2)/win_area*277.8
    
    #---annual total energy cost [$/ft²]---
    annual_cost1 = elec_total*0.2778*float(dfC['Electricity ($/kWh)'].loc[(dfC['State']==st.session_state.states)]) + \
        gas_total*0.2778*float(dfC['Natural Gas ($/kWh)'].loc[(dfC['State']==st.session_state.states)])*0.0929
    annual_cost2 = elec_total2*0.2778*float(dfC['Electricity ($/kWh)'].loc[(dfC['State']==st.session_state.states)]) + \
        gas_total2*0.2778*float(dfC['Natural Gas ($/kWh)'].loc[(dfC['State']==st.session_state.states)])*0.0929
    #---life time cost [$/ft²]---
    life_cost1 = annual_cost1*yr
    life_cost2 = annual_cost2*yr

    if st.session_state.glass1 == 'Triple pane':
        gc1 = 91
    else: 
        gc1 = 59
    if st.session_state.glass2 == 'Triple pane':
        gc2 = 91
    else:
        gc2 = 59
    frame_dens1 = float(dfP['kg/m²'].loc[(dfP['Product name']==st.session_state.prod1)])
    frame_gwp1 = float(dfP['kgCO2/kg'].loc[(dfP['Product name']==st.session_state.prod1)])
    em_carbon1 = gc1 + frame_dens1*frame_gwp1
    if st.session_state.win2:
        frame_dens2 = float(dfP['kg/m²'].loc[(dfP['Product name']==st.session_state.prod2)])
        frame_gwp2 = float(dfP['kgCO2/kg'].loc[(dfP['Product name']==st.session_state.prod2)])
        em_carbon2 = gc2 + frame_dens2*frame_gwp2

  #---average energy to carbon factor [kgCO2/MWh]---
    cf = 400
    em_energy1 = em_carbon1/cf*1000
    em_energy2 = em_carbon2/cf*1000

    total_carbon1 = op_carbon1 + em_carbon1
    total_carbon2 = op_carbon2 + em_carbon2
    st.text("")
    st.header('Fenestration Carbon Emissions (kgCO2/m²)')
    st.text("")
    with st.expander(label='', expanded=False):
        fs = 6
        
        bar1_y1 = np.array([em_carbon1,op_carbon1,total_carbon1])
        bar1_y2 = np.array([em_carbon2,op_carbon2,total_carbon2])
        colors = ['#69A761', '#617165','#85A993','#8C9078']
        bar_labels1 = ["Embodied", "Operational", "Total"]
        labels1 = ["Embodied Carbon", "Operational Carbon"]
        x_axis1 = np.arange(3)
        #col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))
        
        col1, col2, col3,= st.columns([1,5,1])
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5), dpi=300)
            #fig.tight_layout(pad=5)
            fig.subplots_adjust(left=0.3,
                bottom=0.2,
                right=0.8,
                top=0.7,
                wspace=0.4,
                hspace=0.8)
            #fig, (ax1,ax2) = plt.subplots(1, 2)
            #ax1(figsize = (2,2))
            #ax1 = plt.subplot2grid((1,2),(0,0))
            plt.rcParams['font.size'] = fs
            bar1a = ax1.bar(x_axis1 - 0.2, bar1_y1, 0.4, color=colors[0], label=str(st.session_state.prod1))
            bar1b = ax1.bar(x_axis1 + 0.2, bar1_y2, 0.4, color=colors[1], label=str(st.session_state.prod2))
            #ax1.xticks(x_axis1 + 0.2,bar_labels1)
            ax1.set_xticks(x_axis1, minor=False)
            ax1.set_xticklabels(bar_labels1, fontsize=fs)
            #ax1.xlabel("Year")
            #ax1.ylabel("Number of people voted")
            #ax1.title("Number of people voted in each year")
            ax1.legend(fontsize=(fs-2))
            ax1.bar_label(bar1a, fmt ='%5.0f', padding=1)
            ax1.bar_label(bar1b, fmt ='%5.0f', padding=1)
            ax1.set_ylabel(ylabel='Carbon Emissions (kgCO2/m²)',fontsize=fs)
            ax1.tick_params(axis='y', which='major', labelsize=fs)
            ax1.axhline(0, color='black', lw=1)

            #ax2(figsize = (2,2))

            bar2_y1 = np.array([elec_carbon1,gas_carbon1])
            bar2_y2 = np.array([elec_carbon2,gas_carbon2])
            bar_labels2 = ["Electricity", "Natural Gas"]
            x_axis2 = np.arange(2)
            #col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))
            #fig, (ax1,ax2) = plt.subplots(1, 2)
            #ax1(figsize = (2,2))
            #ax1 = plt.subplot2grid((1,2),(0,0))
            bar2a = ax2.bar(x_axis2 - 0.2, bar2_y1, 0.4, color=colors[2], label=str(st.session_state.prod1))
            bar2b = ax2.bar(x_axis2 + 0.2, bar2_y2, 0.4, color=colors[3], label=str(st.session_state.prod2))
            #ax1.xticks(x_axis1 + 0.2,bar_labels1)
            ax2.set_xticks(x_axis2, minor=False)
            ax2.set_xticklabels(bar_labels2, fontsize=fs)
            #ax1.xlabel("Year")
            #ax1.ylabel("Number of people voted")
            #ax1.title("Number of people voted in each year")
            ax2.legend(fontsize=(fs-2))
            ax2.tick_params(axis='y', which='major', labelsize=fs)
            ax2.set_ylabel(ylabel='Carbon Emissions (kgCO2/m²)',fontsize=fs)
            ax2.bar_label(bar2a, fmt ='%5.0f', padding=1)
            ax2.bar_label(bar2b, fmt ='%5.0f', padding=1)
            ax2.axhline(0, color='black', lw=1)
            st.pyplot(fig)

        # col1edge, col1, col2, colmid, col3, col4, col2edge = st.columns((1, 3, 3, 1, 3, 3, 1))
        # col1.subheader(str(st.session_state.prod1))
        # if st.session_state.win2:
        #     col3.subheader(str(st.session_state.prod2))
        # col1.metric('Total carbon footprint 1',"{:.0f}".format(total_carbon1))
        # if st.session_state.win2:
        #     pct = 100*(total_carbon2-total_carbon1)/total_carbon1
        #     col3.metric('Total carbon footprint 2',"{:.0f}".format(total_carbon2),"{:.0f}".format(pct)+'%')
        # col1.text('')
        # col1.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # #col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col4.text('')
        # col1.metric('Operational carbon 1',"{:.0f}".format(op_carbon1))
        # if st.session_state.win2:
        #     pct = 100*(op_carbon2-op_carbon1)/op_carbon1
        #     col3.metric('Operational carbon 2',"{:.0f}".format(op_carbon2),"{:.0f}".format(pct)+'%')
        # col1.text('')
        # col2.metric('Embodied carbon 1',"{:.0f}".format(em_carbon1))
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # col2.text('')
        # if st.session_state.win2:
        #     pct = 100*(em_carbon2-em_carbon1)/em_carbon1
        #     col4.metric('Embodied carbon 2',"{:.0f}".format(em_carbon2),"{:.0f}".format(pct)+'%')
        # col1.text('')
        # col1.text('')
        # col1.text('')
        # col3.text('')
        # col3.text('')
        # col1.metric('Electricity footprint 1',"{:.0f}".format(elec_carbon1))
        # if st.session_state.win2:
        #     pct = 100*(elec_carbon2-elec_carbon1)/elec_carbon1
        #     col3.metric('Electricity footprint 2',"{:.0f}".format(elec_carbon2),"{:.0f}".format(pct)+'%')
        # col1.text('')
        # col2.metric('Natural gas footprint 1',"{:.0f}".format(gas_carbon1))
        # col4.text('')
        # col4.text('')
        # if st.session_state.win2:
        #     pct = 100*(gas_carbon2-gas_carbon1)/gas_carbon1
        #     col4.metric('Natural gas footprint 2',"{:.0f}".format(gas_carbon2),"{:.0f}".format(pct)+'%')
        # col1.text('')

    st.text("")
    st.header('Carbon Graphs')
    st.text("")
    with st.expander(label='', expanded = False):
        col1edge, col1, col2edge = st.columns((1, 4, 1))
        with col1:
            st.subheader('Carbon Footprint Breakdown')
            st.text('')
            if op_carbon1 < 0:
                op_carbon1 = 0
            if op_carbon2 < 0:
                op_carbon2 = 0
            pie_data1 = np.array([em_carbon1,op_carbon1])
            colors = ['#69A761', '#617165','#85A993','#8C9078']
            labels1 = ["Embodied Carbon", "Operational Carbon"]
            #col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (5,5), dpi = 300)
            #ax1(figsize = (2,2))
            #ax1 = plt.subplot2grid((1,2),(0,0))
            ax1.pie(pie_data1, autopct = '%1.0f%%', colors = colors, textprops={'fontsize': 6})
            ax1.set_title(label=str(st.session_state.prod1))
            #st.pyplot(fig)
            ax1.legend(labels1, loc = 'upper left', fontsize = 4)
            
            

            #ax2(figsize = (2,2))
            pie_data2 = np.array([em_carbon2,op_carbon2])
            #ax1 = plt.subplot2grid((2, 4), (0, 1))
            #labels2 = ["Electricity", "Natural Gas"]
            ax2.pie(pie_data2, autopct = '%1.0f%%', colors = colors, textprops={'fontsize': 6})
            ax2.set_title(label=str(st.session_state.prod2))
            ax2.legend(labels1, loc = 'upper right', fontsize = 4)
            st.pyplot(fig)

            t = np.zeros((round(yr+1),1))
            c1 = np.zeros((round(yr+1),1))
            c2 = np.zeros((round(yr+1),1))
            for i in range(yr+1):
                t[i] = i
                if i == 1:
                    c1[i] = em_carbon1
                    c2[i] = em_carbon2
                elif i > 1:
                    c1[i] = c1[i-1] + op_annual1
                    c2[i] = c2[i-1] + op_annual2

            fig3, ax3 = plt.subplots(figsize = (10,5), dpi = 300)
            fs3 = 12
            labels3 = [st.session_state.prod1, st.session_state.prod2]
            ax3.plot(t, c1, linewidth=2.0, color = colors[0])
            ax3.plot(t, c2, linewidth=2.0, color = colors[1])
            ax3.set_xlabel('Time (yrs)', fontsize=fs3)
            ax3.set_ylabel('Cumulative carbon (kgCO2/m²)', fontsize=fs3)
            ax3.legend(labels3, fontsize=(fs3-2))
            #ax1.set_ylabel(ylabel='Carbon Emissions (kgCO2/m²)',fontsize=fs)
            ax3.tick_params(axis='both', which='major', labelsize=fs3)
            #ax3.tick_params(axis='y', which='major', labelsize=fs)
            
            #ax3.set(xlim=(0, 8), xticks=np.arange(1, 8),
            #ylim=(0, 8), yticks=np.arange(1, 8))
            st.pyplot(fig3)

            st.text("")
            st.text("")
            #st.dataframe(df) 
    
    st.header('Fenestration Energy Consumption and Operating Cost')
    st.text("")
    
    with st.expander(label='',expanded=False):
        fs = 6
        rot = 45
        bar4_y1 = np.array([annual_kwh1,annual_kwh2])
        colors = ['#69A761', '#617165','#85A993','#8C9078']
        bar_labels4 = [str(st.session_state.prod1), str(st.session_state.prod2)]
        labels1 = ["Embodied Carbon", "Operational Carbon"]
        x_axis4 = np.arange(2)
        col1, col2, col3,= st.columns([1,5,1])
        with col2:
            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize = (10,5), dpi=300)
            #fig.tight_layout(pad=5)
            fig4.subplots_adjust(left=0.3,
                bottom=0.2,
                right=0.8,
                top=0.7,
                wspace=0.4,
                hspace=0.8)
            #fig, (ax1,ax2) = plt.subplots(1, 2)
            #ax1(figsize = (2,2))
            #ax1 = plt.subplot2grid((1,2),(0,0))
            plt.rcParams['font.size'] = fs
            bar4a = ax4.bar(x_axis4, bar4_y1, 0.4, color=colors[0])
            #bar5b = ax1.bar(x_axis4 + 0.2, bar1_y2, 0.4, color=colors[1], label=str(st.session_state.prod2))
            #ax1.xticks(x_axis1 + 0.2,bar_labels1)
            ax4.set_xticks(x_axis4, minor=False)
            ax4.set_xticklabels(bar_labels4, fontsize=fs, rotation=rot)
            #ax1.legend(fontsize=(fs-2))
            ax4.bar_label(bar4a, fmt ='%5.0f', padding=1)
            #ax4.bar_label(bar4b, fmt ='%5.0f', padding=1)
            ax4.set_ylabel(ylabel='Annual energy [kWh/m²]',fontsize=fs)
            ax4.tick_params(axis='y', which='major', labelsize=fs)
            ax4.axhline(0, color='black', lw=1)

            #ax2(figsize = (2,2))

            bar5_y1 = np.array([annual_cost1,annual_cost2])
            #bar2_y2 = np.array([elec_carbon2,gas_carbon2])
            bar_labels5 = [str(st.session_state.prod1), str(st.session_state.prod2)]
            x_axis5 = np.arange(2)
            #col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))
            #fig, (ax1,ax2) = plt.subplots(1, 2)
            #ax1(figsize = (2,2))
            #ax1 = plt.subplot2grid((1,2),(0,0))
            bar5a = ax5.bar(x_axis5, bar5_y1, 0.4, color=colors[2])
            #bar2b = ax2.bar(x_axis2 + 0.2, bar2_y2, 0.4, color=colors[3], label=str(st.session_state.prod2))
            #ax1.xticks(x_axis1 + 0.2,bar_labels1)
            ax5.set_xticks(x_axis5, minor=False)
            ax5.set_xticklabels(bar_labels5, fontsize=fs, rotation=rot)
            #ax1.xlabel("Year")
            #ax1.ylabel("Number of people voted")
            #ax1.title("Number of people voted in each year")
            ax5.legend(fontsize=(fs-2))
            ax5.tick_params(axis='y', which='major', labelsize=fs)
            ax5.set_ylabel(ylabel='Annual Operating Cost [$/ft²]',fontsize=fs)
            ax5.bar_label(bar5a, fmt ='%5.0f', padding=1)
            ax5.axhline(0, color='black', lw=1)
            #ax5.bar_label(bar2b, fmt ='%5.0f', padding=1)
            st.pyplot(fig4)

        col1edge, col1, colmid, col3, col2edge = st.columns((2, 3, 1, 3, 2))
        col1.subheader(str(st.session_state.prod1))
        if st.session_state.win2:
            col3.subheader(str(st.session_state.prod2))
        col1.metric('Annual energy [kWh/m²]',"{:.1f}".format(annual_kwh1))
        if st.session_state.win2:
            pct = 100*(annual_kwh2-annual_kwh1)/annual_kwh1
            col3.metric('Annual energy [kWh/m²]',"{:.1f}".format(annual_kwh2),"{:.0f}".format(pct)+'%')
        col1.text('')
        col1.text('')
        col1.metric('Energy difference 1',"{:.2f}".format(annual_kwh_diff1))
        col3.metric('Energy difference 2',"{:.2f}".format(annual_kwh_diff2))

        col1.metric('Annual cost [$/ft²]',"{:.2f}".format(annual_cost1))
        if st.session_state.win2:
            diff = annual_cost2 - annual_cost1
            col3.metric('Annual cost [$/ft²]',"{:.2f}".format(annual_cost2),"{:.2f}".format(diff)+'')
        col1.text('')
        col1.text('')
        col1.text('')
        col1.text('')
        col3.text('')
        col3.text('')
        col1.metric('Lifetime cost [$/ft²]',"{:.2f}".format(life_cost1))
        if st.session_state.win2:
            diff = life_cost2 - life_cost1
            col3.metric('Lifetime cost [$/ft²]',"{:.2f}".format(life_cost2),"{:.2f}".format(diff)+'')
        col1.text('')
        col1.text('')
        
    st.header('Payback Period and Return on Investment')
    st.text("")
    with st.expander(label='', expanded=False):
        epb = (em_energy2-em_energy1)/(annual_kwh1-annual_kwh2)
        cpb =  (em_carbon2-em_carbon1)/(op_annual1-op_annual2)
        fpb = float(st.session_state.cost)/(annual_cost1-annual_cost2)

        fs = 6
        rot = 45
        bar6_y1 = np.array([epb,cpb, fpb])
        colors = ['#69A761', '#617165','#85A993','#8C9078']
        bar_labels6 = ['Energy', 'Carbon', 'Financial']
        x_axis6 = np.arange(3)
        col1, col2, col3,= st.columns([1,5,1])
        with col2:
            fig6, ax6 = plt.subplots(figsize = (10,5), dpi=300)
            #fig.tight_layout(pad=5)
            fig6.subplots_adjust(left=0.3,
                bottom=0.2,
                right=0.8,
                top=0.7,
                wspace=0.4,
                hspace=0.8)
            
            plt.rcParams['font.size'] = fs
            bar6a = ax6.bar(x_axis6, bar6_y1, 0.4, color=colors[0])
            
            ax6.set_xticks(x_axis6, minor=False)
            ax6.set_xticklabels(bar_labels6, fontsize=fs, rotation=rot)
            ax6.bar_label(bar6a, fmt ='%5.0f', padding=1)
            ax6.set_ylabel(ylabel='Payback (yrs)',fontsize=fs)
            ax6.tick_params(axis='y', which='major', labelsize=fs)
            ax6.axhline(0, color='black', lw=1)

            st.pyplot(fig6)

        col1edge, col1, col2, col3, col2edge = st.columns((1, 4, 4, 4, 1))
        col1.subheader('ENERGY')
        col2.subheader('CARBON')
        col3.subheader('FINANCIAL')
        col1.metric('Payback Period (yrs)',"{:.2f}".format(epb))
        col2.metric('Payback Period (yrs)',"{:.2f}".format(cpb))
        col3.metric('Payback Period (yrs)',"{:.2f}".format(fpb))
        col1.metric('Return on Investment (%)', "{:.0f}".format(100*(((annual_kwh1-annual_kwh2)*yr-(em_energy2-em_energy1))/(em_energy2-em_energy1))))
        col2.metric('Return on Investment (%)', "{:.0f}".format(100*(total_carbon1-total_carbon2)/(em_carbon2-em_carbon1)))
        col3.metric('Return on Investment (%)', "{:.0f}".format(100*((life_cost1-life_cost2)-float(st.session_state.cost))/(float(st.session_state.cost))))
    
    st.text(elec_blg_total)
    st.text(elec_blg_total2)
    st.text(gas_blg_total)
    st.text(gas_blg_total2)
    st.text(win_area)
    st.button(label='Save PDF',key='pdf') 
    #st.text_input(label='Filename',key="pdf_file")
    #if st.session_state.pdf:
    #    pdfkit.from_url(['http://localhost:8501'], 'D:\my_test_pdf.pdf')
    #return

#def change_prod():

st.set_page_config(layout="wide")
#st.image('D:\Dropbox\OBE\Python\Deploy\logo.jpg')
with st.container():
    col1edge, col1, col2edge = st.columns((1, 12, 1))
    col1.title('OBE Carbon Calculator')
    col1.caption('Calculate the carbon footprint of a fenestration system for commercial buildings across the US')
    col1.caption('')

    
#with st.form(key='my_form'):
#fig.show()
with st.container():
    #col1, col2 = st.columns(2)
    col1edge, col1, col2, col3, col4, col2edge = st.columns((1, 3, 3, 3, 3, 1))

    col1.text_input("Building Floor Area (sq ft)", value = '25000', key="area")

    col2.selectbox('Building Type', ('Office','School','Restaurant','Retail','Apartment','Hotel'),\
        index=0, key='blg', help=None, on_change=None, args=None, kwargs=None, disabled=False)

    col3.selectbox('Building Location (State)', states,\
        index=43, key='states', help=None, on_change=None, args=None, kwargs=None, disabled=False)

    cities = dfW['City'].loc[(st.session_state.states == dfW['State'])]
    col4.selectbox('Building Location', cities,\
        index=0, key='city', help=None, on_change=None, args=None, kwargs=None, disabled=False)

st.caption('')
with st.container():
    #col1, col2 = st.columns(2)
    col1edge, col1, col2, col3, col2edge = st.columns((1, 4, 4, 4, 1))

    col1.slider(label='Number of Floors', min_value=1,max_value=10, value=3, step = 1, key='flr_slider')

    col2.slider(label='Window to Wall Ratio', min_value=0.1,max_value=0.4, value=0.25, key='wwr_slider')

    col3.slider(label='Lifetime (yrs)', min_value=1,max_value=40, value=30, key='yr_slider')
    
    col1.text('')
    col1.text('')
    col1.checkbox(label='Compare 2 Products', key='win2', value=True)
    
    col2.text('')
    col2.text('')
    col2.checkbox(label='Include financial', key='financial')
    col3.number_input(label='Incremental product #2 cost [$/ft]', value = 1.00, min_value = -10.0, max_value = 30.0, step = 0.1, format = "%.2f", key = 'cost')
    
st.caption('')
with st.container():
    #col2.text('')
    #col2.text('')
    col1edge, col1, colmid, col2, col2edge = st.columns((1, 3, 1, 3, 1))

    col1.subheader('Product #1 Details')
    col2.subheader('Product #2 Details')

    col1.selectbox('Product Type', ('Generic U-value','Curtainwall', 'Window wall', 'Store front'),\
        index=0, key='type1', help=None, on_change=None, args=None, kwargs=None, disabled=False)
    if st.session_state.type1:
        frames = dfP.loc[(dfP['Product type']==st.session_state.type1)]
        if st.session_state.type1 == 'Generic U-value':
            uval1_lbl = 'Total product U-value'
        else:
            uval1_lbl = 'COG U-value' 
    if st.session_state.win2:
        col2.selectbox('Product Type', ('Generic U-value','Curtainwall', 'Window wall', 'Store front'),\
            index=0, key='type2', help=None, on_change=None, args=None, kwargs=None, disabled=False)
        if st.session_state.type2:
            frames2 = dfP.loc[(dfP['Product type']==st.session_state.type2)]

            if st.session_state.type2 == 'Generic U-value':
                uval2_lbl = 'Total product U-value'
            else:
                uval2_lbl = 'COG U-value' 
        
    col1.selectbox('OBE Product', frames,\
        index=0, key='prod1', help=None, on_change=None, args=None, kwargs=None, disabled=False)
        
    if st.session_state.win2:
        col2.selectbox('OBE Product', frames2,\
        index=0, key='prod2', help=None, on_change=None, args=None, kwargs=None, disabled=False)

    col1.radio('Glass Type (for embodied carbon only)',('Double pane','Triple pane'), key='glass1')
    if st.session_state.win2:
        col2.radio('Glass Type (for embodied carbon only)',('Double pane','Triple pane'), key='glass2')

    col1.checkbox(label='Manually enter embodied carbon', key='man_ec1')
    if st.session_state.man_ec1:
        col1.number_input(label='Total GWP, frame + glass [kgCO2/m²]', value = 150, min_value = 0, max_value = 1000, step = 10, format = "%i", key = 'ec1')

    if st.session_state.win2:
        col2.checkbox(label='Manually enter embodied carbon', key='man_ec2')
        if st.session_state.man_ec2:
            col2.number_input(label='Total GWP, frame + glass [kgCO2/m²]', value = 150, min_value = 0, max_value = 1000, step = 10, format = "%i", key = 'ec2')   

    if st.session_state.type1 == 'Generic U-value':
        u_max1 = 0.8
    else:
        u_max1 = 0.45
    col1.slider(label=uval1_lbl, min_value=0.1, max_value=u_max1, value=0.30, key='u1_slider')

    if st.session_state.win2:
        if st.session_state.type2 == 'Generic U-value':
            u_max2 = 0.8
        else:
            u_max2 = 0.45
        col2.slider(label=uval2_lbl, min_value=0.1, max_value=u_max2, value=0.30, key='u2_slider')

    col1.slider(label='SHGC', min_value=0.1, max_value=0.8, value=0.35, key='s1_slider')

    if st.session_state.win2:
        col2.slider(label='SHGC', min_value=0.1, max_value=0.8, value=0.35, key='s2_slider')
        if st.session_state.type2 == 'Generic U-value':
            u_total2 = st.session_state.u2_slider
        else:
            u_total2 = dfU[str(st.session_state.prod2)].loc[(dfU['COG U-Factor']==float(st.session_state.u2_slider))]

    if st.session_state.type1 == 'Generic U-value':
        u_total1 = st.session_state.u1_slider
    else:
        u_total1 = dfU[str(st.session_state.prod1)].loc[(dfU['COG U-Factor']==float(st.session_state.u1_slider))]
    col1.metric(label='Total U Value (Btu/hr-°F-ft²)',value = u_total1)
    
    if st.session_state.win2:
        col2.metric(label='Total U Value (Btu/hr-°F-ft²)',value = u_total2)

    st.text("")
    st.text("")
    st.caption('OBE advanced technology - powered by machine learning')

button_calc = st.button(label='Calculate', key ='calc')

if st.session_state.calc:
    run_calc(dfW, dfP)
    #st.write(st.session_state.city)


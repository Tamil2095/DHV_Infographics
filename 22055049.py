# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 19:20:32 2023

@author: TAMILSELVAN
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


def process_dataframe(df, series_name, countries):
    '''Process the dataframe for the given series name and countries.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - series_name (str): Name of the series to filter.
    - countries (list): List of country names.

    Returns:
    - drop_series (pd.DataFrame): Processed dataframe with selected series and countries.
    - df_t (pd.DataFrame): Transposed dataframe for further analysis.
    '''
    data = df.rename(columns=dict((col, col.split()[0]) for col in df.columns[4:]))
    data_sel = data[data['Series Name'] == series_name]
    cln_data_series = data_sel.reset_index(drop=True)
    drop_series = cln_data_series.drop(['Country Code', 'Series Name', 'Series Code'], axis=1)

    df_t = drop_series.transpose()
    df_t.columns = df_t.iloc[0]
    df_t = df_t.iloc[1:]
    df_t.index = pd.to_numeric(df_t.index)
    df_t['Years'] = df_t.index
    df_t.reset_index(drop=True)
    
    return drop_series, df_t


def plot(data1, data2, data3, data4, report_text, author_name, student_id):
    '''Generate plots and save the final report with author information.

    Parameters:
        - data1, data2, data3, data4 (pd.DataFrame): Dataframes for plotting.
        - report_text (str): Text for the report.
        - author_name (str): Author's name.
        - student_id (str): Student ID.
        - github (str): GitHub link.

    '''
    # Set the global style for seaborn
    sns.set_style("whitegrid")

    # Create a 3x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9), gridspec_kw={'hspace': 0.4}, facecolor='lavender')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.1, wspace=0.2, hspace=0.2)

    # Plot 1: Bar Plot
    ax1 = axes[0, 0]
    df_filtered = data1[data1['Years'].isin([2000, 2010, 2020])]
    df_filtered.plot(x='Years', y=['Belgium', 'Estonia', 'Finland', 'Ireland', 'Norway'], kind='bar', ax=ax1, 
                     width=0.65, xlabel='Years', ylabel='billions of people')
    ax1.set_title('Total Population of People', fontsize=14)
    ax1.set_facecolor('lavender')
    ax1.legend(loc='best', bbox_to_anchor=(1, 0.4), facecolor='lavender')

    # Plot 2: Donut Plot
    ax2 = axes[0, 1]
    label = data2['Country Name'].tolist()
    values = data2['2020'].tolist()
    outer_pie = ax2.pie(values, labels=None, startangle=90, wedgeprops=dict(width=0.3, edgecolor='w'))
    center_circle = plt.Circle((0, 0), 0.35, fc='lavender')
    ax2.add_artist(center_circle)
    ax2.axis('equal')
    ax2.set_title(f'Freshwater Withdrawals % for Agriculture in 2020', fontsize=14)
    ax2.set_facecolor('lavender')
    legend_labels = [f'{label[i]}: {values[i]:1.2f}%' for i in range(len(label))]
    ax2.legend(outer_pie[0], legend_labels, bbox_to_anchor=(0.8, 0.5), loc="center left", fontsize=10, facecolor='lavender')

    # Plot 3: Horizontal Bar Plot
    ax3 = axes[1, 0]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    sns.barplot(x='2020', y='Country Name', data=data3, palette=colors, ax=ax3)
    ax3.set(xlabel='% of Withdrawals', ylabel='Countries')
    ax3.set_facecolor('lavender')
    ax3.set_title('Freshwater withdrawals % for Domestic in 2020 year', fontsize=14)
    for rect in ax3.patches:
        width = rect.get_width()
        ax3.text(width, rect.get_y() + rect.get_height() / 2, f'{width:.2f}', ha='left', va='center', weight='bold')

    # Plot 4: Line Plot
    ax4 = axes[1, 1]
    linePlotData = data4[['Years', 'Belgium', 'Estonia', 'Finland', 'Ireland', 'Norway']]
    linePlotData.plot(x='Years', kind='line', marker='.', ax=ax4)
    ax4.set_title('Freshwater withdrawals % for Industry ', fontsize=14)
    ax4.set_xlabel('Years')
    ax4.set_xticks(range(2000, 2021, 4))
    ax4.set_ylabel('% of Withdrawals')
    ax4.set_facecolor('lavender')
    ax4.legend(loc='best', bbox_to_anchor=(1, 0.4), facecolor='lavender')

    # Wrap the report text
    text = textwrap.fill(report_text, width=145, initial_indent='', subsequent_indent='')

    # Text style for the report
    text_style = {'fontsize': 16, 'fontstyle': 'normal', 'color': 'black'}

    # Plot the report text
    plt.text(0.55, -0.06, text, transform=fig.transFigure, ha="center", va="center",
             bbox=dict(boxstyle='round', facecolor='#ADD8E6', alpha=0.2, edgecolor='darkgray'), fontdict=text_style)

    # Stylish box for author information
    author_box = dict(boxstyle='round', facecolor='#FFD700', alpha=0.5, edgecolor='black')

    # Plot author information
    plt.text(0.80, -0.20, f"Author: {author_name}", transform=fig.transFigure, ha="left", va="bottom", fontsize=14, color='black', bbox=author_box)
    plt.text(0.80, -0.24, f"Student id: {student_id}", transform=fig.transFigure, ha="left", va="bottom", fontsize=14, color='black', bbox=author_box)
    
    plt.suptitle("Population Growth and Water Usage Trends (2000-2020)", fontsize=22, y=0.95, color='white', ha='center', backgroundcolor='black')

    plt.savefig('22055049.png', dpi=300, bbox_inches='tight')
    
    
# Read the CSV file
df = pd.read_csv('dhv_file3.csv')
country_list = ['Belgium', 'Estonia', 'Finland', 'Ireland', 'Norway']

# Process dataframes
df1, df1_t = process_dataframe(df, 'Population, total', country_list)
df2, df2_t = process_dataframe(df, 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)', country_list)
df3, df3_t = process_dataframe(df, 'Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)', country_list)
df4, df4_t = process_dataframe(df, 'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)', country_list)

# Report text
report_text = """
This report explores population growth and water usage trends over two decades for selected European countries. Belgium led with 11.5 million in 2020, while Norway and Ireland showed 
growth, and Estonia and Finland remained stable. In 2020, Belgium's agricultural water usage rose by 1.27%, Estonia peaked at 0.57%,and Finland at 28.57%. Ireland's usage stabilized at 2.46%, and 
Norway slightly declined but stabilized at 31.39%. Domestic water usage in 2020 saw Belgium at 17.33%, Estonia spiking to 7.38%, Finland stable at 14.29%, Ireland increasing to 63.96% in 2020, and
Norway steady at 28.81%. The line plot reveals Belgium consistently led industrial water usage at 81.39%, Estonia fluctuated but reached 92.05% in 2020, Finland stayed around 57.14%, Ireland 
declined to 33.58% in 2020, and Norway maintained around 39.80% from 2000 to 2020. In conclusion, population growth impacts water usage in the selected countries, with Belgium leading in industrial and agricultural sectors, and Ireland experiencing a notable rise in domestic water 
consumption. These trends emphasize the need for strategic water resource management.

"""

author_name = "TAMILSELVAN PALANISAMY"
student_id = "22055049"

plot(df1_t, df2, df3, df4_t, report_text, author_name, student_id)
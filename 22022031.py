#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 03:51:15 2024

@author: diya
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_path = 'most-polluted-countries.csv'
pollution_data = pd.read_csv(file_path)

colormap = plt.cm.RdYlGn.reversed()
norm = plt.Normalize(vmin=pollution_data[
    'mostPollutedCountries_particlePollution'].min(),
                     vmax=pollution_data[
                         'mostPollutedCountries_particlePollution'].max())
scalar_map = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
scalar_map.set_array([])
background_color = 'silver'
sns.set(style="whitegrid", palette="pastel", context='talk')


plt.rcParams.update({
    'font.size': 10,
    'axes.titleweight': 'bold',
    'font.family': 'Comic Sans MS',
    'axes.labelweight': 'bold',
    'figure.facecolor': background_color,
    'axes.facecolor': background_color,
    'savefig.facecolor': background_color,
})

#plt.figure(figsize=(30, 20))
fig, axs = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={
                        'height_ratios': [2, 2], 'width_ratios': [2, 2],
                        'hspace': 0.7})
plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9,
                    top=0.85, wspace=0.4, hspace=0.4)


def plot_particle_pollution(data, specific_countries, scalar_map, ax):
    """
    Plot a horizontal bar chart of particle pollution for specified countries.

    Parameters:
    data (DataFrame): The pollution data.
    specific_countries (list): List of countries to be plotted.
    scalar_map (ScalarMappable): Scalar mappable object for color mapping.
    ax (Axes): The axes object to plot on.
    """
    data_specific = data[data['country_name'].isin(specific_countries)]
    data_specific_sorted = data_specific.sort_values(
        by='mostPollutedCountries_particlePollution', ascending=False)
    pollution_values = data_specific_sorted[
        'mostPollutedCountries_particlePollution']
    bar_colors = scalar_map.to_rgba(pollution_values)
    bars = ax.barh(
        data_specific_sorted['country_name'], pollution_values,
        color=bar_colors)
    ax.set_xlabel('Particle Pollution (µg/m³)', fontsize=12, labelpad=40)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Particle Pollution for Specified Countries',
                 fontsize=13, pad=7)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.invert_yaxis()
    cbar = plt.colorbar(scalar_map, ax=ax)
    cbar.set_label('Particle Pollution (µg/m³)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.grid(True, which='both', axis='both',
            color='gray', linestyle='-', linewidth=0.2)


def plot_pollution_growth_and_density(data, ax1):
    """
    Plot a combined line and bar chart showing the pollution growth rate and density per mile by region.

    Parameters:
    data (DataFrame): The pollution data.
    ax1 (Axes): The primary axes object for the bar plot.
    """
    avg_pollution_growth_by_region = data.groupby(
        'country_region')['pollution_growth_Rate'].mean()
    avg_pollution_density_by_region = data.groupby(
        'country_region')['pollution_density_per_Mile'].mean()
    sorted_regions = avg_pollution_growth_by_region.sort_values().index
    ax1.fill_between(sorted_regions, avg_pollution_density_by_region.reindex(
        sorted_regions), color='darkgreen', alpha=0.5,
        label='Pollution Density per Mile')
    ax2 = ax1.twinx()
    ax2.plot(sorted_regions,
             avg_pollution_growth_by_region.reindex(sorted_regions),
             marker='o', linestyle='-', color='red',
             label='Pollution Growth Rate')
    ax1.set_xlabel('Region', fontsize=12, labelpad=2)
    ax1.set_ylabel('Average Pollution Density per Mile',
                   fontsize=12, color='darkgreen')
    ax1.tick_params(axis='x', labelsize=10, rotation=90)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.set_ylabel('Average Pollution Growth Rate', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelsize=10)
    ax1.set_title(
        'Average Pollution Growth Rate and Density per Mile by Region',
        fontsize=13, pad=7)
    ax1.grid(True, which='both', axis='both',
             color='gray', linestyle='-', linewidth=0.2)


def plot_pollution_across_regions(data, ax):
    """
    Plot a pie chart showing the percentage of pollution produced across different regions.

    Parameters:
    data (DataFrame): The pollution data.
    ax (Axes): The axes object to plot on.
    """
    region_pollution = data.groupby('country_region')['pollution_2023'].sum()
    region_pollution_percent = (
        region_pollution / region_pollution.sum()) * 100
    explode = [0.04] * len(region_pollution_percent)
    wedges, texts, autotexts = ax.pie(region_pollution_percent,
                                      autopct='%1.1f%%', startangle=200,
                                      colors=plt.cm.RdYlGn(np.linspace(
                                          0, 1, len(region_pollution_percent))),
                                      explode=explode,
                                      wedgeprops=dict(width=0.7),
                                      pctdistance=1.2)
    ax.set_title('Percentage of Pollution produced across Regions',
                 fontsize=15, pad=10)
    ax.text(0, 0, '2023', ha='center', va='center',
            size=10, color="black", weight='bold')
    for text in texts:
        text.set_fontsize(10)
    legend = ax.legend(wedges, region_pollution_percent.index, title="Region",
                       loc="center left",
                       bbox_to_anchor=(1.1, 0.5), fontsize=10)
    legend.get_title().set_fontsize(12)
    legend.get_title().set_fontweight('bold')


def plot_comparative_regional_analysis(data, ax):
    """
    Plot a bar and line chart comparing the average land area and pollution density per km for regions.

    Parameters:
    data (DataFrame): The pollution data.
    ax (Axes): The primary axes object for the bar plot.
    """
    region_grouped = data.groupby('country_region').agg({
        'country_land_Area_in_Km': 'mean',
        'pollution_density_in_km': 'mean'
    }).reset_index()
    colors = colormap(np.linspace(0, 1, len(region_grouped)))
    bars = ax.bar(region_grouped['country_region'],
                  region_grouped['country_land_Area_in_Km'], color=colors)
    ax.set_xlabel('Region', fontsize=12, labelpad=1)
    ax.set_ylabel('Average Land Area in Km', color='tab:blue', fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, which='both', axis='y',
            color='gray', linestyle='-', linewidth=0.1)
    ax2 = ax.twinx()
    ax2.plot(region_grouped['country_region'],
             region_grouped['pollution_density_in_km'], color='red',
             marker='o')
    ax2.set_ylabel('Average Pollution Density (per Km)',
                   color='tab:red', fontsize=12)
    # Increase font size for y-axis tick labels
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(True, which='both', axis='both',
             color='gray', linestyle='-', linewidth=0.4)
    ax.set_title('Comparative Regional Analysis', fontsize=15, pad=10)


# Specify the countries of interest and call the plotting functions
specific_countries = ['India', 'China', 'United States',
                      'Australia', 'Japan', 'Iceland', 'Pakistan']
plot_particle_pollution(
    pollution_data, specific_countries, scalar_map, axs[0, 0])
plot_pollution_growth_and_density(pollution_data, axs[0, 1])
plot_pollution_across_regions(pollution_data, axs[1, 0])
plot_comparative_regional_analysis(pollution_data, axs[1, 1])

# Set the main title and add description text
plt.suptitle('Global Pollution Analysis - 2023',
             fontsize=20, weight='bold', ha='center')

# Define a box style for highlighting with a specific color
student_details_params = {
    'facecolor': 'snow',  # light blue
    'alpha': 0.7,
    'edgecolor': 'black',
    'boxstyle': 'round,pad=1'
}

# Add the description text and other plotting configurations
Plot_description = """

Comprehensive Overview of Global Particle Pollution:

Particle Pollution for Specified Countries:
 * Highlights stark contrasts in air quality, with Pakistan at a high of 60 µg/m³ and Iceland at a low of 10 µg/m³.

Pollution Growth and Density Comparison by Region:
 * Illustrates diverse environmental impacts with variations in pollution density and growth rates across regions.

Percentage of Pollution Produced Across Regions (2023):
 * Asia dominates with a 65.3% share, indicating regional imbalances in global pollution output.

Comparative Regional Analysis:
 * Reveals that pollution density is not directly proportional to land area, with Europe showing higher pollution levels despite smaller land area compared to Africa.

The data underscores the urgent need for targeted environmental policies, particularly in Asia, to address the stark regional disparities and high pollution levels that do not correlate directly with land size.


"""

plt.figtext(0.2, -0.3, Plot_description, ha='left',
            va='center', fontsize=15, wrap=True, weight='bold')

# Add highlighted name and student ID
student_details = "Name       : Vinoth Rajendran\nStudent ID : 22022031"
plt.figtext(0.2, -0.01, student_details, ha='left', va='center',
            fontsize=12, weight='bold', bbox=student_details_params)

# plt.tight_layout(pad=4.0)

plt.subplots_adjust(top=0.90, bottom=0.1)
#plt.savefig("22022031.png", dpi=300, bbox_inches='tight')
plt.show()  # Display the canvas with all subplots

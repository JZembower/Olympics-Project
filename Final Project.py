import pandas as pd
import numpy as np
from matplotlib import pylab as pyl

# Read in the data set
olympics = pd.read_csv('/Users/jonahzembower/OneDrive - Seton Hill University/Coding Applications/Datasets/ProgTwoMathAndEngineering/OlympicsData/athlete_events.csv')
olympics.head()
print(olympics.isnull().sum())
olympics['Medal'].fillna('DNW', inplace=True)
# As expected the NaNs in the 'Medal' column disappear!
print(olympics.isnull().sum())
print(olympics.loc[:, ['NOC', 'Team']].drop_duplicates()['NOC'].value_counts().head())
# Lets read in the noc_country mapping first
noc_country = pd.read_csv('/Users/jonahzembower/OneDrive - Seton Hill University/Coding Applications/Datasets/ProgTwoMathAndEngineering/OlympicsData/noc_regions.csv')
noc_country.drop('notes', axis=1, inplace=True)
noc_country.rename(columns={'region': 'Country'}, inplace=True)

noc_country.head()

# merging
olympics_merge = olympics.merge(noc_country,
                                left_on='NOC',
                                right_on='NOC',
                                how='left')

# Do we have NOCs that didn't have a matching country in the master?
olympics_merge.loc[olympics_merge['Country'].isnull(), ['NOC', 'Team']].drop_duplicates()

# Replace missing Teams by the values above.
olympics_merge.loc[olympics_merge['Country'].isnull(), ['Country']] = olympics_merge['Team']

olympics_merge['Country'] = np.where(olympics_merge['NOC'] == 'SGP', 'Singapore', olympics_merge['Country'])
olympics_merge['Country'] = np.where(olympics_merge['NOC'] == 'ROT', 'Refugee Olympic Athletes',
                                     olympics_merge['Country'])
olympics_merge['Country'] = np.where(olympics_merge['NOC'] == 'UNK', 'Unknown', olympics_merge['Country'])
olympics_merge['Country'] = np.where(olympics_merge['NOC'] == 'TUV', 'Tuvalu', olympics_merge['Country'])

# Put these values from Country into Team
olympics_merge.drop('Team', axis=1, inplace=True)
olympics_merge.rename(columns={'Country': 'Team'}, inplace=True)

# Let's take data from 1961 onwards only and for summer olympics only
olympics_complete = olympics_merge
olympics_complete_subset = olympics_complete.loc[
                           (olympics_complete['Year'] > 1960) & (olympics_complete['Season'] == "Summer"), :]

# Reset row indices
olympics_complete_subset = olympics_complete_subset.reset_index()

olympics_complete_subset['Medal_Won'] = np.where(olympics_complete_subset.loc[:, 'Medal'] == 'DNW', 0, 1)

# Check whether number of medals won in a year for an event by a team exceeds 1. This indicates a team event.
identify_team_events = pd.pivot_table(olympics_complete_subset,
                                      index=['Team', 'Year', 'Event'],
                                      columns='Medal',
                                      values='Medal_Won',
                                      aggfunc='sum',
                                      fill_value=0).drop('DNW', axis=1).reset_index()

identify_team_events = identify_team_events.loc[identify_team_events['Gold'] > 1, :]

team_sports = identify_team_events['Event'].unique()

remove_sports = ["Gymnastics Women's Balance Beam", "Gymnastics Men's Horizontal Bar",
                 "Swimming Women's 100 metres Freestyle", "Swimming Men's 50 metres Freestyle"]

team_sports = list(set(team_sports) - set(remove_sports))

# if an event name matches with one in team sports, then it is a team event. Others are singles events.
team_event_change = olympics_complete_subset['Event'].map(lambda x: x in team_sports)
single_event_change = [not i for i in team_event_change]

# rows where medal_won is 1
medal_change = olympics_complete_subset['Medal_Won'] == 1

# Put 1 under team event if medal is won and event in team event list
olympics_complete_subset['Team_Event'] = np.where(team_event_change & medal_change, 1, 0)

# Put 1 under singles event if medal is won and event not in team event list
olympics_complete_subset['Single_Event'] = np.where(single_event_change & medal_change, 1, 0)

# Add an identifier for team/single event
olympics_complete_subset['Event_Category'] = olympics_complete_subset['Single_Event'] + \
                                             olympics_complete_subset['Team_Event']

medal_tally_irrelevant = olympics_complete_subset. \
    groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']]. \
    agg('sum').reset_index()

medal_tally_irrelevant['Medal_Won_Corrected'] = medal_tally_irrelevant['Medal_Won'] / medal_tally_irrelevant[
    'Event_Category']

# Medal Tally.
medal_tally = medal_tally_irrelevant.groupby(['Year', 'Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

medal_tally_pivot = pd.pivot_table(medal_tally,
                                   index='Team',
                                   columns='Year',
                                   values='Medal_Won_Corrected',
                                   aggfunc='sum',
                                   margins=True).sort_values('All', ascending=False)[1:5]

# print total medals won in the given period
medal_tally_pivot.loc[:, 'All']

# List of top countries
top_countries = ['USA', 'Russia', 'Germany', 'China']

year_team_medals = pd.pivot_table(medal_tally,
                                  index='Year',
                                  columns='Team',
                                  values='Medal_Won_Corrected',
                                  aggfunc='sum')[top_countries]

# plotting the medal tallies
year_team_medals.plot(linestyle='-', marker='o', alpha=0.9, figsize=(10, 8), linewidth=2)
pyl.xlabel('Olympic Year')
pyl.ylabel('Number of Medals')
pyl.title('Olympic Performance Comparison')

# List of top countries
top_countries = ['USA', 'Russia', 'Germany', 'China']

# row change where countries match
row_change_2 = medal_tally_irrelevant['Team'].map(lambda x: x in top_countries)

# Pivot table to calculate sum of gold, silver and bronze medals for each country
medal_tally_specific = pd.pivot_table(medal_tally_irrelevant[row_change_2],
                                      index=['Team'],
                                      columns='Medal',
                                      values='Medal_Won_Corrected',
                                      aggfunc='sum',
                                      fill_value=0).drop('DNW', axis=1)

# Re-order the columns so that they appear in order on the chart
medal_tally_specific = medal_tally_specific.loc[:, ['Gold', 'Silver', 'Bronze']]

medal_tally_specific.plot(kind='bar', stacked=True, figsize=(8, 6), rot=0)
pyl.xlabel('Number of Medals')
pyl.ylabel('Country')

# To get the sports, teams are best at, we now aggregate the medal_tally_irrelevant dataframe as we did earlier.
best_team_sports = pd.pivot_table(medal_tally_irrelevant[row_change_2],
                                  index=['Team', 'Event'],
                                  columns='Medal',
                                  values='Medal_Won_Corrected',
                                  aggfunc='sum',
                                  fill_value=0).sort_values(['Team', 'Gold'], ascending=[True, False]).reset_index()

best_team_sports.drop(['Bronze', 'Silver', 'DNW'], axis=1, inplace=True)
best_team_sports.columns = ['Team', 'Event', 'Gold_Medal_Count']

best_team_sports.groupby('Team').head(5)

# take for each year, the team, name of the athlete and gender of the athlete and drop duplicates. These are values
# where the same athlete is taking part in more than one sport.

# get rows with top countries
row_change_3 = olympics_complete_subset['Team'].map(lambda x: x in top_countries)

year_team_gender = olympics_complete_subset.loc[row_change_3, ['Year', 'Team', 'Name', 'Sex']].drop_duplicates()

# Create a pivot table to count gender wise representation of each team in each year
year_team_gender_count = pd.pivot_table(year_team_gender,
                                        index=['Year', 'Team'],
                                        columns='Sex',
                                        aggfunc='count').reset_index()

# rename columns as per column names in the 0th level
year_team_gender_count.columns = year_team_gender_count.columns.get_level_values(0)

# rename the columns appropriately
year_team_gender_count.columns = ['Year', 'Team', 'Female_Athletes', 'Male_Athletes']

# get total athletes per team-year
year_team_gender_count['Total_Athletes'] = year_team_gender_count['Female_Athletes'] + \
                                           year_team_gender_count['Male_Athletes']

# Separate country wise data

chi_data = year_team_gender_count[year_team_gender_count['Team'] == "China"]
chi_data.fillna(0, inplace=True)
chi_data.set_index('Year', inplace=True)

ger_data = year_team_gender_count[year_team_gender_count['Team'] == "Germany"]
ger_data.set_index('Year', inplace=True)

rus_data = year_team_gender_count[year_team_gender_count['Team'] == "Russia"]
rus_data.set_index('Year', inplace=True)

usa_data = year_team_gender_count[year_team_gender_count['Team'] == "USA"]
usa_data.set_index('Year', inplace=True)

# Plot the values of male, female and total athletes using bar charts and the line charts.
fig, ((ax1, ax2), (ax3, ax4)) = pyl.subplots(nrows=2, ncols=2, figsize=(20, 12), sharey=True)
fig.subplots_adjust(hspace=0.3)

# Plot team China's contingent size
ax1.bar(chi_data.index.values, chi_data['Male_Athletes'], width=-1, align='edge', label='Male Athletes')
ax1.bar(chi_data.index.values, chi_data['Female_Athletes'], width=1, align='edge', label='Female Athletes')
ax1.plot(chi_data.index.values, chi_data['Total_Athletes'], linestyle=':', color='black', label='Total Athletes',
         marker='o')
ax1.set_title('Team China:\nComposition over the years')
ax1.set_ylabel('Number of Athletes')
ax1.legend(loc='best')

# Plot team USA's contingent size
ax2.bar(usa_data.index.values, usa_data['Male_Athletes'], width=-1, align='edge', label='Male Athletes')
ax2.bar(usa_data.index.values, usa_data['Female_Athletes'], width=1, align='edge', label='Female Athletes')
ax2.plot(usa_data.index.values, usa_data['Total_Athletes'], linestyle=':', color='black', label='Total Athletes',
         marker='o')
ax2.set_title('Team USA:\nComposition over the years')
ax2.set_ylabel('Number of Athletes')
ax2.legend(loc='best')

# Plot team Germany's contingent size
ax3.bar(ger_data.index.values, ger_data['Male_Athletes'], width=-1, align='edge', label='Male Athletes')
ax3.bar(ger_data.index.values, ger_data['Female_Athletes'], width=1, align='edge', label='Female Athletes')
ax3.plot(ger_data.index.values, ger_data['Total_Athletes'], linestyle=':', color='black', label='Total Athletes',
         marker='o')
ax3.set_title('Team Germany:\nComposition over the years')
ax3.set_ylabel('Number of Athletes')
ax3.legend(loc='best')

# Plot team Russia's contingent size
ax4.bar(rus_data.index.values, rus_data['Male_Athletes'], width=-1, align='edge', label='Male Athletes')
ax4.bar(rus_data.index.values, rus_data['Female_Athletes'], width=1, align='edge', label='Female Athletes')
ax4.plot(rus_data.index.values, rus_data['Total_Athletes'], linestyle=':', color='black', label='Total Athletes',
         marker='o')
ax4.set_title('Team Russia:\nComposition over the years')
ax4.set_ylabel('Number of Athletes')
ax4.legend(loc='best')

pyl.show()

# Get year wise team wise athletes.
year_team_athelete = olympics_complete_subset.loc[row_change_3, ['Year', 'Team', 'Name']].drop_duplicates()

# sum these up to get total contingent size.
contingent_size = pd.pivot_table(year_team_athelete,
                                 index='Year',
                                 columns='Team',
                                 values='Name',
                                 aggfunc='count')

fig, ((ax1, ax2), (ax3, ax4)) = pyl.subplots(nrows=2,
                                             ncols=2,
                                             figsize=(20, 12))

fig.subplots_adjust(hspace=0.3)

# Plot China's medal tally and contingent size
contingent_size['China'].plot(ax=ax1, linestyle='-', marker='o', linewidth=2, color='red',
                              label='Contingent Size')
year_team_medals['China'].plot(ax=ax1, linestyle='-', marker='o', linewidth=2, color='black',
                               label='Medal Tally')
ax1.plot(2008, contingent_size.loc[2008, 'China'], marker='^', color='red', ms=14)
ax1.plot(2008, year_team_medals.loc[2008, 'China'], marker='^', color='black', ms=14)
ax1.set_xlabel('Olympic Year')
ax1.set_ylabel('Number of Athletes/Medal Tally')
ax1.set_title('Team China\nContingent Size vs Medal Tally')
ax1.legend(loc='best')

# Plot USA's medal tally and contingent size
contingent_size['USA'].plot(ax=ax2, linestyle='-', marker='o', linewidth=2, color='blue',
                            label='Contingent Size')
year_team_medals['USA'].plot(ax=ax2, linestyle='-', marker='o', linewidth=2, color='black',
                             label='Medal Tally')
ax2.plot(1984, contingent_size.loc[1984, 'USA'], marker='^', color='blue', ms=14)
ax2.plot(1984, year_team_medals.loc[1984, 'USA'], marker='^', color='black', ms=14)
ax2.set_xlabel('Olympic Year')
ax2.set_ylabel('Number of Athletes/Medal Tally')
ax2.set_title('Team USA\nContingent Size vs Medal Tally')
ax2.legend(loc='best')

# Plot Germany's medal tally and contingent size
contingent_size['Germany'].plot(ax=ax3, linestyle='-', marker='o', linewidth=2, color='green',
                                label='Contingent Size')
year_team_medals['Germany'].plot(ax=ax3, linestyle='-', marker='o', linewidth=2, color='black',
                                 label='Medal Tally')
ax3.plot(1972, year_team_medals.loc[1972, 'Germany'], marker='^', color='black', ms=14)
ax3.plot(1972, contingent_size.loc[1972, 'Germany'], marker='^', color='green', ms=14)
ax3.set_xlabel('Olympic Year')
ax3.set_ylabel('Number of Athletes/Medal Tally')
ax3.set_title('Team Germany\nContingent Size vs Medal Tally')
ax3.legend(loc='best')

# Plot Russia's medal tally and contingent size
contingent_size['Russia'].plot(ax=ax4, linestyle='-', marker='o', linewidth=2, color='orange',
                               label='Contingent Size')
year_team_medals['Russia'].plot(ax=ax4, linestyle='-', marker='o', linewidth=2, color='black',
                                label='Medal Tally')
ax4.plot(1980, contingent_size.loc[1980, 'Russia'], marker='^', color='orange', ms=14)
ax4.plot(1980, year_team_medals.loc[1980, 'Russia'], marker='^', color='black', ms=14)
ax4.set_xlabel('Olympic Year')
ax4.set_ylabel('Number of Athletes/Medal Tally')
ax4.set_title('Team Russia\nContingent Size vs Medal Tally')
ax4.legend(loc='best')

pyl.show()

# Lets merge contingent size and medals won!
year_team_medals_unstack = year_team_medals.unstack().reset_index()
year_team_medals_unstack.columns = ['Team', 'Year', 'Medal_Count']

contingent_size_unstack = contingent_size.unstack().reset_index()

contingent_size_unstack.columns = ['Team', 'Year', 'Contingent']

contingent_medals = contingent_size_unstack.merge(year_team_medals_unstack,
                                                  left_on=['Team', 'Year'],
                                                  right_on=['Team', 'Year'])

contingent_medals[['Contingent', 'Medal_Count']].corr()

# merge best team sports with olympics data to get sport for each event.
team_commonalities = best_team_sports.merge(olympics_complete_subset.loc[:, ['Sport', 'Event']].drop_duplicates(),
                                            left_on='Event',
                                            right_on='Event')

team_commonalities = team_commonalities.sort_values(['Team', 'Gold_Medal_Count'], ascending=[True, False])
team_commonalities = team_commonalities.groupby('Team').head(5).reset_index()

# make a pivot table of the commonalities.
team_commonalities = pd.pivot_table(team_commonalities,
                     index='Sport',
                     columns='Team',
                     values='Event',
                     aggfunc='count',
                     fill_value=0,
                     margins=True).sort_values('All', ascending=False)[1:]

olympics_complete_subset[['Year', 'City']].drop_duplicates().sort_values('Year')

# Correct city names in the dataset
olympics_complete_subset['City'].replace(['Athina', 'Moskva'], ['Athens', 'Moscow'], inplace=True)

# city to country mapping dictionary
city_to_country = {'Tokyo': 'Japan',
                   'Mexico City': 'Mexico',
                   'Munich': 'Germany',
                   'Montreal': 'Canada',
                   'Moscow': 'Russia',
                   'Los Angeles': 'USA',
                   'Seoul': 'South Korea',
                   'Barcelona': 'Spain',
                   'Atlanta': 'USA',
                   'Sydney': 'Australia',
                   'Athens': 'Greece',
                   'Beijing': 'China',
                   'London': 'UK',
                   'Rio de Janeiro': 'Brazil'}

# Map cities to countries
olympics_complete_subset['Country_Host'] = olympics_complete_subset['City'].map(city_to_country)

# print the
olympics_complete_subset.loc[:, ['Year', 'Country_Host']].drop_duplicates().sort_values('Year')

# Extract year, host nation and team name from the data
year_host_team = olympics_complete_subset[['Year', 'Country_Host', 'Team']].drop_duplicates()

# check rows where host country is the same as team
row_change_4 = (year_host_team['Country_Host'] == year_host_team['Team'])

# add years in the year_host_team to capture one previous and one later year
year_host_team['Prev_Year'] = year_host_team['Year'] - 4
year_host_team['Next_Year'] = year_host_team['Year'] + 4

# Subset only where host nation and team were the same
year_host_team = year_host_team[row_change_4]

# Calculate the medals won in each year where a team played at home. merge year_host_team with medal_tally on year and team
year_host_team_medal = year_host_team.merge(medal_tally,
                                            left_on=['Year', 'Team'],
                                            right_on=['Year', 'Team'],
                                            how='left')

year_host_team_medal.rename(columns={'Medal_Won_Corrected': 'Medal_Won_Host_Year'}, inplace=True)

# Calculate medals won by team in previous year
year_host_team_medal = year_host_team_medal.merge(medal_tally,
                                                  left_on=['Prev_Year', 'Team'],
                                                  right_on=['Year', 'Team'],
                                                  how='left')

year_host_team_medal.drop('Year_y', axis=1, inplace=True)
year_host_team_medal.rename(columns={'Medal_Won_Corrected': 'Medal_Won_Prev_Year',
                                     'Year_x': 'Year'}, inplace=True)

# Calculate the medals won by the team the year after they hosted.
year_host_team_medal = year_host_team_medal.merge(medal_tally,
                                                  left_on=['Next_Year', 'Team'],
                                                  right_on=['Year', 'Team'],
                                                  how='left')

year_host_team_medal.drop('Year_y', axis=1, inplace=True)
year_host_team_medal.rename(columns={'Year_x': 'Year',
                                     'Medal_Won_Corrected': 'Medal_Won_Next_Year'}, inplace=True)

# General formatting changes
year_host_team_medal.drop(['Prev_Year', 'Next_Year'], axis=1, inplace=True)
year_host_team_medal.sort_values('Year', ascending=True, inplace=True)
year_host_team_medal.reset_index(inplace=True, drop=True)

# column re-ordering
year_host_team_medal = year_host_team_medal.loc[:,
                       ['Year', 'Country_Host', 'Team', 'Medal_Won_Prev_Year', 'Medal_Won_Host_Year',
                        'Medal_Won_Next_Year']]

# lets create a data frame of athletes with the sport they participated in and the number of medals won.
ath_sport_medal = olympics_complete_subset.groupby(['Team', 'Name', 'Sport'])['Medal_Won'].agg('sum').reset_index()
ath_sport_medal.sort_values(['Sport', 'Medal_Won'], ascending=[True, False], inplace=True)

# keep only athletes who won medals
medal_change = ath_sport_medal['Medal_Won'] > 0
ath_sport_medal = ath_sport_medal[medal_change]

ath_sport_medal.head()

# Now lets calculate the number of participations of each athlete. This will be sport wise.
ath_sport_appearance = olympics_complete_subset.groupby(['Team', 'Name', 'Sport'])['NOC'].agg('count').reset_index()

ath_sport_appearance.rename(columns={'NOC': 'Event_Count'}, inplace=True)

ath_sport_appearance.head()

# lets merge these two.
ath_medal_appearance = ath_sport_medal.merge(ath_sport_appearance,
                                             left_on=["Team", "Name", "Sport"],
                                             right_on=['Team', 'Name', 'Sport'],
                                             how="left")

# Calculate the medal per participation
ath_medal_appearance['Medal_Per_Participation'] = ath_medal_appearance['Medal_Won'] / ath_medal_appearance[
    'Event_Count']

ath_medal_appearance.sort_values(['Medal_Per_Participation', 'Medal_Won'], ascending=[False, False], inplace=True)

ath_medal_appearance.head(10)

# filter out athletes with less than 10 total medals.
ath_medal_appearance = ath_medal_appearance[ath_medal_appearance['Medal_Won'] >= 10]

# create the year, team contingent size
year_team_gender = olympics_complete_subset.loc[:, ['Year', 'Team', 'Name', 'Sex']].drop_duplicates()

year_team_gender_count = pd.pivot_table(year_team_gender,
                                        index=['Year', 'Team'],
                                        columns='Sex',
                                        aggfunc='count').reset_index()

# rename columns as per column names in the 0th level
year_team_gender_count.columns = year_team_gender_count.columns.get_level_values(0)

# rename the columns appropriately
year_team_gender_count.columns = ['Year', 'Team', 'Female_Athletes', 'Male_Athletes']
year_team_gender_count = year_team_gender_count.fillna(0)

# get total athletes per team-year
year_team_gender_count['Total_Athletes'] = year_team_gender_count['Female_Athletes'] + \
                                           year_team_gender_count['Male_Athletes']

year_team_contingent = year_team_gender_count.loc[:, ['Year', 'Team', 'Total_Athletes']]
year_team_contingent.head()

medal_tally.head()

# get host nation from the data
year_host = olympics_complete_subset.loc[:, ['Year', 'Country_Host']].drop_duplicates()

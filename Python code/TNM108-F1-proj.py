#Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from array import array

#Datasets
circuits = pd.read_csv('./F1Data/circuits.csv', index_col='circuitId')
constructor_results = pd.read_csv('./F1Data/constructor_results.csv', index_col='constructorResultsId')
constructor_standings = pd.read_csv('./F1Data/constructor_standings.csv', index_col='constructorStandingsId')
constructors = pd.read_csv('./F1Data/constructors.csv', index_col='constructorId')
driver_standings = pd.read_csv('./F1Data/driver_standings.csv', index_col='driverStandingsId')
drivers = pd.read_csv('./F1Data/drivers.csv', index_col='driverId')
lap_times = pd.read_csv('./F1Data/lap_times.csv', index_col='raceId')
pit_stops = pd.read_csv('./F1Data/pit_stops.csv', index_col='raceId')
qualifying = pd.read_csv('./F1Data/qualifying.csv', index_col='qualifyId')
races = pd.read_csv('./F1Data/races.csv', index_col='raceId')
results = pd.read_csv('./F1Data/results.csv', index_col='resultId')
seasons = pd.read_csv('./F1Data/seasons.csv', index_col='year')
sprint_results = pd.read_csv('./F1Data/sprint_results.csv', index_col='resultId')
status = pd.read_csv('./F1Data/status.csv', index_col='statusId')

# Profiles
profile1 = {
  'FavDriver': 'Max Verstappen',
  'FavConst' : 'Red Bull',
  'FavCircuit' : 'Red Bull Ring',
  'FavRace'  : 'Belgian Grand Prix 2022'
}
profile2 = {
  'FavDriver': 'Michael Schumacher',
  'FavConst' : 'Ferrari',
  'FavCircuit' : 'Hockenheimring',
  'FavRace'  : 'Belgian Grand Prix 1998'
}
profile3 = {
  'FavDriver': 'Fernando Alonso',
  'FavConst' : 'Renault',
  'FavCircuit' : 'Circuit de Barcelona-Catalunya',
  'FavRace'  : 'Malaysian Grand Prix 2004'
}
profile4 = {
  'FavDriver': 'Lewis Hamilton',
  'FavConst' : 'Mercedes',
  'FavCircuit' : 'Silverstone Circuit',
  'FavRace'  : 'Turkish Grand Prix 2020'
}
profile5 ={
  'FavDriver': 'Valtteri Bottas',
  'FavConst' : 'Sauber',
  'FavCircuit' : 'Autódromo José Carlos Pace',
  'FavRace'  : 'Monaco Grand Prix 1996'
}

profiles = [profile1,profile2,profile3, profile4,profile5] # Collect profiles to array

# Functions
def createProfiles(profiles): # Takes the array profiles, containing all profiles
    profileDf = pd.DataFrame()
    indexCounter = 1                # Used in the subsequent for loop
    # Convert profile parameters to relevant indices
    for i in profiles:
    # Driver
        FavoriteDriver = i['FavDriver'].split(' ',1)
        driverMask = (drivers['forename'].str.contains(FavoriteDriver[0])) & (drivers['surname'].str.contains(FavoriteDriver[1]))
        foundDriver = drivers[driverMask]
        profileDf.loc[indexCounter, 'driverId'] = int(foundDriver.index[0])
        profileDf.loc[indexCounter, 'driverName'] = foundDriver['forename'].iloc[0] + " " + foundDriver['surname'].iloc[0]
        profileDf.loc[indexCounter, 'driverDob'] = foundDriver['dob'].iloc[0]

        # Constructor
        FavoriteConst = i['FavConst']
        constMask = (constructors['name'].str.contains(FavoriteConst))
        foundConstructor = constructors[constMask]
        profileDf.loc[indexCounter, 'constructorId'] = int(foundConstructor.index[0])
        profileDf.loc[indexCounter, 'constructorName'] =  foundConstructor['name'].iloc[0]
        profileDf.loc[indexCounter, 'constructorCountry'] = foundConstructor['nationality'].iloc[0]

        # Circuit
        FavoriteCircuit = i['FavCircuit']
        circuitMask = (circuits['name'].str.contains(FavoriteCircuit))
        foundCircuit = circuits[circuitMask]
        profileDf.loc[indexCounter, 'circuitId'] = int(foundCircuit.index[0])
        profileDf.loc[indexCounter, 'circuitName'] = foundCircuit['name'].iloc[0]
        profileDf.loc[indexCounter, 'circuitCountry'] = foundCircuit['country'].iloc[0]

        # Race
        FavoriteRaceText = i['FavRace'].rstrip('0123456789')
        FavortieRaceText = FavoriteRaceText.rstrip('Grand Prix')
        FavoriteRaceYear = i['FavRace'][len(FavoriteRaceText):]
        raceMask = (races['name'].str.contains(FavortieRaceText) & (races['year'] == int(FavoriteRaceYear)))
        foundRace = races[raceMask]
        profileDf.loc[indexCounter, 'raceId'] = int(foundRace.index[0])
        profileDf.loc[indexCounter, 'raceYear'] = foundRace['year'].iloc[0]
        profileDf.loc[indexCounter, 'raceName'] = foundRace['name'].iloc[0]

        # Advance counter
        indexCounter = indexCounter + 1
    return profileDf
def filterRaces():
    # Filter time and unimportant dates
    FilteredRaces = races.drop(columns=['time','fp1_date','fp1_time','fp2_date','fp2_time','fp3_date', 'fp3_time', 'quali_date', 'quali_time','sprint_date','sprint_time'])
    
    # Add qualifying and race order for all races. Includes drivers and constructors
    driversQuali = qualifying.groupby('raceId',group_keys=True)['driverId'].apply(list)
    constructorsQuali = qualifying.groupby('raceId',group_keys=True)['constructorId'].apply(list)
    driversPosition = results.groupby('raceId', group_keys=True)['driverId'].apply(list)
    constructorsPosition = results.groupby('raceId',group_keys=True)['constructorId'].apply(list)

    # Add new columns for qualifying and race order to FilteredRaces
    FilteredRaces['driverQuali'] = driversQuali
    FilteredRaces['constructorsQuali'] = constructorsQuali
    FilteredRaces['driversPosition'] = driversPosition
    FilteredRaces['constructorsPosition'] = constructorsPosition

    return FilteredRaces
def calcCosineSimularity(raceDf):
    driverCosim=[]
    constrCosim=[]

    for i in races.index:
        if (not raceDf['driverQuali'].isnull().loc[i]) & (not raceDf['driversPosition'].isnull().loc[i]):
            if len(raceDf['driverQuali'].loc[i]) == len(raceDf['driversPosition'].loc[i]):
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i]).reshape(1,-1), np.array(raceDf['driversPosition'].loc[i]).reshape(1,-1)))
            elif len(raceDf['driverQuali'].loc[i]) > len(raceDf['driversPosition'].loc[i]):
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i])[: len(raceDf['driversPosition'].loc[i])].reshape(1,-1), np.array(raceDf['driversPosition'].loc[i]).reshape(1,-1)))
            else:
                driverCosim.append(cosine_similarity(np.array(raceDf['driverQuali'].loc[i]).reshape(1,-1), np.array(raceDf['driversPosition'].loc[i])[: len(raceDf['driverQuali'].loc[i])].reshape(1,-1)))
        else:
            driverCosim.append(np.nan)

        if (not raceDf['constructorsQuali'].isnull().loc[i]) & (not raceDf['constructorsPosition'].isnull().loc[i]):
            if len(raceDf['constructorsQuali'].loc[i]) == len(raceDf['constructorsPosition'].loc[i]):
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i]).reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i]).reshape(1,-1)))
            elif len(raceDf['constructorsQuali'].loc[i]) > len(raceDf['constructorsPosition'].loc[i]):
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i])[: len(raceDf['constructorsPosition'].loc[i])].reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i]).reshape(1,-1)))
            else:
                constrCosim.append(cosine_similarity(np.array(raceDf['constructorsQuali'].loc[i]).reshape(1,-1), np.array(raceDf['constructorsPosition'].loc[i])[: len(raceDf['constructorsQuali'].loc[i])].reshape(1,-1)))
        else:
            constrCosim.append(np.nan)

    raceDf['driverCosim'] = driverCosim
    raceDf['constrCosim'] = constrCosim


    return raceDf
def calcDriverClimb(raceDf):
    podiumClimb = []

    for i in raceDf.index:
        if (not raceDf['driverQuali'].isnull().loc[i]) & (not raceDf['driversPosition'].isnull().loc[i]):
            podiumClimbRace = []

            for driverIndex, driver in enumerate(raceDf['driversPosition'].loc[i]):
                try:
                    endPosition = driverIndex
                    startPosition = raceDf['driverQuali'].loc[i].index(driver)
                    positionDiff = startPosition - endPosition
                    podiumClimbRace.append([driver, positionDiff])
                except:
                    podiumClimbRace.append(np.nan)
                    continue
            podiumClimb.append(podiumClimbRace)
        else:
            podiumClimb.append(np.nan)
    
    raceDf['podiumClimb'] = podiumClimb
    return raceDf

# Main

profDf = createProfiles(profiles) # Generate the profiles dataframe
filtRaces = filterRaces()               # Filter races.cvs and add relevant information
filtRaces = calcCosineSimularity(filtRaces)
filtRaces = calcDriverClimb(filtRaces)

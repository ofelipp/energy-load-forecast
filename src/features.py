# ! ./venv/bin/python3.8

"""
Script destinated to create fetures dataframes used as input time-series
Neural Network model
"""

import pandas as pd
import numpy as np
import ephem
import datetime

from src.data import ibge_datetime
from src.io import list_dir_files, json_to_dict

ONS_DATA = "data/raw/ons/"
EXP_DATA = "data/prc/"
PIB_DATA = "data/raw/ibge/pib/"
IND_DATA = "data/raw/ibge/industria/"
POP_DATA = "data/raw/ibge/populacao/"

SEASON_RANGES = "src/static/season_ranges.json"

IND_SECTORS = {
    '1 Indústria geral': "General",
    '2 Indústrias extrativas': "Extractive",
    '3 Indústrias de transformação': "Transformation"
}

IND_PRODUCTS = {
    '1 Bens de capital': "Bens_Capital",
    '2 Bens intermediários': "Bens_Intermediario",
    '31 Bens de consumo duráveis': "Bens_Consumo_Duraveis",
    '32 Bens de consumo semiduráveis e não duráveis': "Bens_Consumo_Nao_Duraveis"
}

MONTH_NAME_NUMBER = {
    "janeiro": "01",
    "fevereiro": "02",
    "março": "03",
    "abril": "04",
    "maio": "05",
    "junho": "06",
    "julho": "07",
    "agosto": "08",
    "setembro": "09",
    "outubro": "10",
    "novembro": "11",
    "dezembro": "12"
}

PIB_SECTORS = {
    'Exportação de bens e serviços': "Export_MM_Reais",
    'Importação de bens e serviços (-)': "Import_MM_Reais"
}

REGIONS_UF = {
    "Rondônia": "Norte",
    "Acre": "Norte",
    "Amazonas": "Norte",
    "Roraima": "Norte",
    "Pará": "Norte",
    "Amapá": "Norte",
    "Tocantins": "Norte",
    "Maranhão": "Nordeste",
    "Piauí": "Nordeste",
    "Ceará": "Nordeste",
    "Rio Grande do Norte": "Nordeste",
    "Paraíba": "Nordeste",
    "Pernambuco": "Nordeste",
    "Alagoas": "Nordeste",
    "Sergipe": "Nordeste",
    "Bahia": "Nordeste",
    "Minas Gerais": "Sudeste",
    "Espírito Santo": "Sudeste",
    "Rio de Janeiro": "Sudeste",
    "São Paulo": "Sudeste",
    "Paraná": "Sul",
    "Santa Catarina": "Sul",
    "Rio Grande do Sul": "Sul",
    "Mato Grosso do Sul": "Centro-Oeste",
    "Mato Grosso": "Centro-Oeste",
    "Goiás": "Centro-Oeste",
    "Distrito Federal": "Centro-Oeste"
}


def datetime_variables(
    series_datetime: pd.Series,
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
) -> pd.DataFrame():

    """
    Receive a Series with Datetime values and returns a DataFrame containing
    Year, Month, Day, Hour and Minute variables.
    """

    # Standarlize Datetime format
    series_datetime = pd.to_datetime(series_datetime, format=datetime_format)

    df_datetime = pd.DataFrame()
    int_vars = []

    # Date variables
    df_datetime["Year"] = series_datetime.dt.strftime("%Y")
    df_datetime["Month"] = series_datetime.dt.strftime("%m")
    df_datetime["Day"] = series_datetime.dt.strftime("%d")
    df_datetime["InverseDay"] = None # TODO
    int_vars += ["Year", "Month", "Day"]

    # Time variables
    df_datetime["Hour"] = series_datetime.dt.strftime("%H")
    df_datetime["Minute"] = series_datetime.dt.strftime("%M")
    int_vars += ["Hour", "Minute"]

    # Week
    df_datetime["WeekDay"] = series_datetime.dt.dayofweek()
    df_datetime["WeekMonth"] = None # TODO
    df_datetime["WeekYear"] = series_datetime.dt.isocalendar().week
    int_vars += ["WeekDay", "WeekMonth", "WeekYear"]

    # Year variable
    df_datetime["YearDay"] = series_datetime.dt.dayofyear()
    int_vars += ["YearDay"]

    # Transforming into integers
    for col in int_vars:
        df_datetime[col] = df_datetime[col].astype(int)

    return df_datetime


def calendar_variables() -> pd.DataFrame():

    """
    Based on a Datetime serie create calendar variables indicating if there
    is a national or regional holiday, week day, year day, month day, etc..
    """

    return ...


def easter_date(start_period: str, end_period: str) -> pd.Series():

    """ Calculates the Easter date between a period of datetimes """

    EQUINOX = ['03-21']

    def next_full_moon_datetime(date: str) -> pd.Series():

        """ Receive datetime and retrieves the next full moon datetime """

        date_tuple = ephem.next_full_moon(date).tuple()

        return pd.to_datetime(
            datetime.datetime(
                year=date_tuple[0], month=date_tuple[1], day=date_tuple[2]
            )
        )

    # Serie containing period
    period = pd.DataFrame({
        "Datetime": pd.date_range(start_period, end_period)
    })

    # Equinox Dates
    years = list(period["Datetime"].dt.strftime("%Y").unique())

    equinox = []

    for year in years:
        equinox += [year + "-" + date for date in EQUINOX]

    equinox = pd.Series(equinox)

    # Next full moon after equinox
    next_full_moon = equinox.apply(
        lambda date: next_full_moon_datetime(date)
    )

    # First Sunday after Equinox with a full moon
    days_left_sunday = pd.to_timedelta(6 - next_full_moon.dt.weekday, 'day')

    return pd.Series(next_full_moon + days_left_sunday)


def carnival_date(easter_dates: pd.Series()) -> pd.Series:

    """
    Returns Carnival dates passing a serie of Easter dates

    Occurs 47 days before Easter, 40 days before Palm Sunday which occurs 7
    days before Easter.

    Carnival + 40 -> Palm Sunday
    Palm Sunday + 7 -> Easter

    """

    return pd.to_datetime(easter_dates) - pd.to_timedelta(47, 'day')


def ashes_wednesday(carnival_dates: pd.Series()) -> pd.Series:

    """
    Returns Ashes Wednesday dates passing a serie of Carnival dates

    Occurs a day after the carnival festival
    """

    return pd.to_datetime(carnival_dates) + pd.to_timedelta(1, 'day')


def saint_friday(easter_dates: pd.Series()) -> pd.Series:

    """
    Returns Saint Friday dates passing a serie of Easter dates

    Occurs in the friday before Easter
    """

    return pd.to_datetime(easter_dates) - pd.to_timedelta(2, 'day')


def season(model_datetime_data: pd.Series) -> np.ndarray:
    """
    Create Season variable using datetime column from pandas DataFrame

    Args:
        model_data = pd.DataFrame : containing datetime col
        datetime_col = str : datetime column name which the process of creation
                       will use.

    """

    # Importing ranges from json
    season_ranges = json_to_dict(SEASON_RANGES)

    # Create function dataframe
    datetime_col = str(model_datetime_data.name)
    df_season = pd.DataFrame({datetime_col: model_datetime_data})
    df_season[datetime_col] = df_season[datetime_col].dt.normalize()

    # Auxiliar columns
    df_season["Season"] = None
    df_season["Year"] = df_season[datetime_col].dt.strftime("%Y")

    # Filling df_season with season info
    for name, period in season_ranges["seasons"].items():
        print(name, period)

        # Creating series with season start date
        df_season[name + "_start"] = pd.to_datetime(
            df_season["Year"] + "-" + period["start"]
        )

        # Creating series with season ending date
        df_season[name + "_end"] = pd.to_datetime(
            df_season["Year"] + "-" + period["end"]
        )

        # Condition
        _starting = df_season[datetime_col] >= df_season[name + "_start"]
        _ending = df_season[datetime_col] <= df_season[name + "_end"]

        # Category
        df_season.loc[_starting & _ending, "Season"] = name[:6]

        # Drop aux seasons columns
        _drop_cols = [name + "_start", name + "_end"]
        df_season.drop(columns=_drop_cols, inplace=True)

    return df_season["Season"].values


def electric_demand(export: bool = True) -> pd.DataFrame():

    '''
    Electric Demand partioned by subregion
    '''

    # Read and concate
    electric_demand = pd.DataFrame()

    for file in list_dir_files(ONS_DATA):
        try:
            print(file)
            demand = pd.read_csv(file, encoding='utf16')
            electric_demand = pd.concat([electric_demand, demand])
        except UnicodeDecodeError:
            print('Erro')

    # Creating time variables
    electric_demand = pd.concat(
        [
            electric_demand,
            datetime_variables(
                series_datetime=electric_demand["Datetime"],
                datetime_format="%d/%m/%Y %H:%M:%S"
            )
        ],
        axis=1
    )

    # Sorting Values
    electric_demand.sort_values(["Datetime", "Subsystem"], inplace=True)

    # Droping Duplicates
    electric_demand.drop_duplicates(inplace=True)

    # Pivoting
    electric_demand = electric_demand.pivot(
        index="Datetime",
        columns="Subsystem",
        values="Demand_MWh"
    ).reset_index()

    # Renaming
    electric_demand.columns = ["Datetime"] + \
        [
            "Demand_" + col + "_MWh"
            for col in electric_demand.columns
            if col not in ["Datetime"]
    ]

    if export:
        electric_demand.to_parquet(
            f"{EXP_DATA}electricity_demand.parquet", index=False
        )

    return electric_demand


def industry_production(type: str, export: bool = True) -> pd.DataFrame():

    """
    Function used to create industry dataset with fisical production percentage
    from different industry sectors (General, Extractive and Transformation).

    Returns a single pd.DataFrame.

    **Extracted**: https://www.ibge.gov.br/estatisticas/economicas/industria/
    9296-pesquisa-industrial-mensal-producao-fisica-regional.html?=&
    t=series-historicas
    """

    if type == "sector":
        _input_path = f"{IND_DATA}200201_202209_pim_pf_tipo_industria.csv"
        _initial_cols = ["Ind_Sector", "Region"]

    elif type == "product":
        _input_path = f"{IND_DATA}200201_202209_pim_pf_tipo_bem_com_ajuste_sazonal.csv"
        _initial_cols = ["Region", "Ind_Product"]

    else:
        print("Error")

    # Read
    industry_raw = pd.read_csv(
        _input_path, sep=';', skiprows=1, skipfooter=1, engine="python"
    )

    # Pre-treatment
    industry_raw.columns = _initial_cols + list(industry_raw.columns[2:])
    _ind_cat = [col for col in _initial_cols if 'Ind' in col]

    industry = industry_raw.melt(
        _ind_cat,
        var_name="YM",
        value_name="PFI_Perc"
    )

    # Datetime
    tmp_ym_industry = industry["YM"].str.split(expand=True)
    tmp_ym_industry.rename(columns={0: "Month", 1: "Year"}, inplace=True)

    industry = pd.concat([industry, tmp_ym_industry], axis=1)

    industry["Month"] = industry["Month"].map(MONTH_NAME_NUMBER)
    industry["Datetime"] = pd.to_datetime(
        industry["Year"] + '-' + industry["Month"]
    )

    industry.drop(columns=["YM", "Month", "Year"], inplace=True)
    industry.dropna(subset="Datetime", inplace=True)

    # Maping Industry Categories
    if type == "sector":
        industry[_ind_cat[0]] = industry[_ind_cat[0]].map(IND_SECTORS)
    elif type == "product":
        industry[_ind_cat[0]] = industry[_ind_cat[0]].map(IND_PRODUCTS)
    else:
        print("Error")

    industry = industry.pivot("Datetime", _ind_cat, "PFI_Perc")
    industry.reset_index(drop=False, inplace=True)

    return industry


def pib_import_export(export: bool = True) -> pd.DataFrame():

    # Read
    pib_raw = pd.read_csv(
        f"{PIB_DATA}1996_2022_PIB_Trimestral_Importacao_Exportacao_MM_Reais.csv",
        sep=';', skiprows=1, skipfooter=1, engine="python"
    )

    # Melting
    _cat_col = pib_raw.columns[0]
    pib = pib_raw.melt(
        _cat_col, var_name="Date", value_name="PIB_MM_Reais"
    )

    # Datetime creation
    pib["Datetime"] = ibge_datetime(pib["Date"], 'trimester')
    pib.dropna(subset="Datetime", inplace=True)

    # Standarlize categories
    pib[_cat_col] = pib[_cat_col].map(PIB_SECTORS)

    pib = pib.pivot("Datetime", _cat_col, "PIB_MM_Reais")
    pib.reset_index(drop=False, inplace=True)

    pib.columns.names = [None]

    if export:
        pib.to_parquet(f"{EXP_DATA}pib_import_export.parquet", index=False)

    return pib


def pib_current_value(export: bool = True) -> pd.DataFrame():

    # Read
    pib_raw = pd.read_csv(
        f"{PIB_DATA}1996_2022_PIB_Trimestral_Valor_Corrente_MM_Reais.csv",
        sep=';', skiprows=1, skipfooter=1, engine="python"
    )

    # Melting
    _cat_col = pib_raw.columns[0]
    pib = pib_raw.melt(
        _cat_col, var_name="Date", value_name="PIB_MM_Reais"
    )

    # Datetime creation
    pib["Datetime"] = ibge_datetime(pib["Date"], 'trimester')
    pib.dropna(subset="Datetime", inplace=True)

    # Standarlize categories
    pib = pib[['Datetime', 'PIB_MM_Reais']].copy()

    if export:
        pib.to_parquet(f"{EXP_DATA}pib_current_value.parquet", index=False)

    return pib


def pib_growth_perc(export: bool = True) -> pd.DataFrame():

    # Read
    pib_raw = pd.read_csv(
        f"{PIB_DATA}1996_2022_PIB_Trimestral_Taxa_Acumulada_Perc.csv",
        sep=';', skiprows=1, skipfooter=1, engine="python"
    )

    # Melting
    _cat_col = pib_raw.columns[0]
    pib = pib_raw.melt(
        _cat_col, var_name="Date", value_name="PIB_Growth_Perc"
    )

    # Datetime creation
    pib["Datetime"] = ibge_datetime(pib["Date"], 'trimester')
    pib.dropna(subset="Datetime", inplace=True)

    # Standarlize categories
    pib = pib[['Datetime', 'PIB_Growth_Perc']].copy()

    if export:
        pib.to_parquet(f"{EXP_DATA}pib_growth.parquet", index=False)

    return pib


def population_projection(export: bool = True) -> tuple:

    """
    Function used to create population dataset features separated in possible
    categories like total, by region or by UF (States).

    Returns three pd.DataFrames with each category.
    """

    pop_raw = pd.read_excel(
        f"{POP_DATA}2010_2030_Populacao_Projetada_Mensal.xlsx",
        skiprows=1, skipfooter=2
    )

    pop = pop_raw.melt("DATA", var_name="UF", value_name="Population_hab")

    pop["Region"] = pop["UF"].map(REGIONS_UF)

    # Separation by categories

    # Brazil total
    _br = pop["UF"] == "Brasil"
    pop_total = pop[_br].copy()
    pop_total.drop(columns="Region", inplace=True)
    pop_total.rename(columns={"DATA": "Datetime", "UF": "Region"}, inplace=True)

    # Regions
    _regions = pop["Region"].isnull()
    pop_region = pop[_regions & ~_br].copy()
    pop_region.drop(columns="Region", inplace=True)
    pop_region.rename(
        columns={"DATA": "Datetime", "UF": "Region"},
        inplace=True
    )

    # Federative states
    pop_uf = pop[~_regions & ~_br].copy()
    pop_uf.rename(columns={"DATA": "Datetime"}, inplace=True)

    if export:
        pop_total.to_parquet(f"{EXP_DATA}pop_total.parquet")
        pop_region.to_parquet(f"{EXP_DATA}pop_region.parquet")
        pop_uf.to_parquet(f"{EXP_DATA}pop_uf.parquet")

    return tuple([pop_total, pop_region, pop_uf])

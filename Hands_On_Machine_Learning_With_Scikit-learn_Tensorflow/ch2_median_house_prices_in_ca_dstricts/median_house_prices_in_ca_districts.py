from sklearn.model_selection import train_test_split
import pandas as pd
import urllib
import tarfile
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"

# Common imports

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(df, test_ratio):
    np.random.seed(RANDOM_STATE)
    shuffled_indicies = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    train_set = df.iloc[train_indicies]
    test_set = df.iloc[test_indicies]
    # print("(Custom) Training size = ", train_set.shape)
    # print("(Custom) Testing size = ", test_set.shape, "\n")
    # print(train_set.head())
    # split using builtin sklearn
    train_set, test_set = train_test_split(
        df, test_size=test_ratio, random_state=RANDOM_STATE)
    # print("(sklearn train_test_split) Training size = ", train_set.shape)
    # print("(sklearn train_test_split) Testing size = ", test_set.shape, "\n")
    # print(train_set.head())
    return train_set, test_set


def explore_data(df):
    print(df.head(), "\n")
    print(df.shape, "\n")
    print(df.info(), "\n")
    print(df["ocean_proximity"].value_counts(), "\n")
    print(df.describe(), "\n")
    # df.hist(bins=50, figsize=(20, 15))
    # plt.show()


def create_income_categories(df):
    df["income_categories"] = pd.cut(df["median_income"],
                                     bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                     labels=[1, 2, 3, 4, 5]
                                     )
    # df["income_categories"].hist()
    # print("income category distribution proportions in original dataframe")
    # print(df["income_categories"].value_counts()/len(df)*100)
    return df

def split_by_strata(df):
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in split.split(housing, housing["income_categories"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    # print("income category distribution proportions in stratified split")
    # print(strat_train_set["income_categories"].value_counts()/len(strat_train_set)*100)
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_categories", axis=1, inplace=True)
    return strat_train_set, strat_test_set

def visualize_data(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    save_fig("better_visualization_plot")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    plt.legend()
    save_fig("housing_prices_scatterplot")

    # Download the California image
    images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
    os.makedirs(images_path, exist_ok=True)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    filename = "california.png"
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
    urllib.request.urlretrieve(url, os.path.join(images_path, filename))

    import matplotlib.image as mpimg
    california_img=mpimg.imread(os.path.join(images_path, filename))
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                        s=housing['population']/100, label="Population",
                        c="median_house_value", cmap=plt.get_cmap("jet"),
                        colorbar=False, alpha=0.4,
                        )
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
            cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)

    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()
    return housing

def explore_correleations(housing, visualize=False):
    corr_matrix = housing.corr()
    # print("correlations matrix")
    # print(corr_matrix, "\n")
    # print(corr_matrix["median_house_value"].sort_values(ascending=False), "\n")
    from pandas.plotting import scatter_matrix
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    if visualize:
        scatter_matrix(housing[attributes], figsize=(20, 16))
        housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(12,8))
        plt.axis([0, 16, 0, 550000])
        save_fig("income_vs_house_value_scatterplot")
    print("adding calculated columns")
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    corr_matrix = housing.corr()
    # print("correlations matrix after adding columns")
    # print(corr_matrix, "\n")
    # print(corr_matrix["median_house_value"].sort_values(ascending=False), "\n")
    if visualize:
        housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
                alpha=0.2)
        plt.axis([0, 5, 0, 520000])
        plt.show()
    return housing

def prepare_data(housing, strat_train_set):
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    # print("null values in our dataframe")
    # print(sample_incomplete_rows.head())

    #lets use imputer class to replace NaN values with median values for that columns
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    ## get all numerical columns
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
    housing_tr.loc[sample_incomplete_rows.index.values]



# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def get_full_pipeline(housing):    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder


    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])    

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
    return full_pipeline

result_columns = ["model_name", "model", "predictions", "mse", "rmse", "mae", "cross_val_rmse_scores",
 "cross_val_mean", "cross_val_std"]
def prepare_result(model, model_name, predictions, mse, rmse, mae, cross_val_rmse_scores):
    cross_val_mean = np.nan
    cross_val_std = np.nan
    if isinstance(cross_val_rmse_scores, np.ndarray):
        cross_val_mean = cross_val_rmse_scores.mean()
        cross_val_std = cross_val_rmse_scores.std()    
    else:
        cross_val_rmse_scores = np.nan

    result = {result_columns[0]:model_name,
            #   result_columns[1]:model,
            #   result_columns[2]:predictions,
              result_columns[3]:mse,
              result_columns[4]:rmse,
              result_columns[5]:mae,
            #   result_columns[6]:cross_val_rmse_scores,
              result_columns[7]:cross_val_mean,
              result_columns[8]:cross_val_std              
    }
    return result

def train_test(model,model_name_str, housing_prepared, housing_labels, housing ):
    from sklearn.linear_model import LinearRegression

    # model = LinearRegression()
    # print(housing_prepared)
    # print(housing_prepared.shape)
    model.fit(housing_prepared, housing_labels)
    housing_predictions = model.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)    
    mae = mean_absolute_error(housing_labels, housing_predictions)

    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    result = prepare_result(model, model_name_str, housing_predictions, mse, rmse, mae, rmse_scores)
    # print(result)
    return result



def train_test_models(housing_prepared, housing_labels, housing):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR

    models = {"LinearRegression":LinearRegression(), 
             "DecisionTreeRegressor":DecisionTreeRegressor(),
             "RandomForestRegressor":RandomForestRegressor(random_state=RANDOM_STATE),
             "SVR":SVR(kernel="linear")}
    # models = {"LinearRegression":LinearRegression()}

    results_df = pd.DataFrame()
    for model_name, model in models.items():
        print("modelling {} - ".format(model_name), end="")
        result = train_test(model, model_name, housing_prepared, housing_labels, housing)
        print("rmse = {}, cross_val_mean = {} cross_val_std = {}".format(result[result_columns[4]], 
                result[result_columns[7]], result[result_columns[8]]))
        results_df = results_df.append([result], ignore_index = True)


    return results_df

    


if __name__ == '__main__':

    # fetch_housing_data()
    housing = load_housing_data() 
    # explore_data(housing)
    
    # get train test split - our own implementation. 
    # train_set, test_set = split_train_test(housing, TEST_SIZE)

    #lets create income categories. This can be used to create additional columns
    housing = create_income_categories(housing)
    
    #split by strata - The real split
    strat_train_set, strat_test_set = split_by_strata(housing)

    # now as far as we are concerned we have forgottend strat_test_set and strat_train_set if our full housing data
    #setup features and lables
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # implemented but not going to do now as it will block the execution
    # housing = visualize_data(housing)
    # housing = explore_correleations(housing)

    #get the full pipepline
    full_pipeline = get_full_pipeline(housing)
    housing_prepared = full_pipeline.fit_transform(housing)

    #now we are ready to run it. 
    results_df = train_test_models(housing_prepared, housing_labels, housing)
    print()
    print(results_df)




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Define the features we'll use for prediction
features = ['income', 'credit_score', 'debt', 'savings', 'employment_length']

def get_user_input():
    """Function to get user input for loan prediction."""
    user_data = {}
    for feature in features:
        user_data[feature] = float(input(f"Enter {feature}: "))
    return user_data

def create_model(X, y):
    """Train a Random Forest Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, accuracy_score(y_test, y_pred)

def evaluate(individual, X, y):
    """Evaluate the fitness of a feature subset."""
    selected_features = [index for index, bit in enumerate(individual) if bit]
    if len(selected_features) == 0:
        return 0.0,
    
    model, accuracy = create_model(X[:, selected_features], y)
    return accuracy,

# Check if classes are already created
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(features))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate, X=np.random.rand(100, len(features)), y=np.random.randint(0, 2, 100))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main execution
def main():
    # Dummy data for demonstration
    X = np.random.rand(100, len(features))
    y = np.random.randint(0, 2, 100)

    # Genetic Algorithm
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, 
                                              stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    best_features = [features[i] for i, bit in enumerate(best_ind) if bit]
    
    print(f"Best feature subset: {best_features}")

    # Use the best features to get user input
    user_input = {feat: float(input(f"Enter {feat}: ")) for feat in best_features}
    
    # Convert user input to numpy array for prediction
    X_user = np.array([user_input[feat] for feat in best_features]).reshape(1, -1)
    
    # Create model with best features
    model, _ = create_model(X[:, [features.index(feat) for feat in best_features]], y)
    
    # Predict
    prediction = model.predict(X_user)
    print(f"Loan approval likelihood: {'Approved' if prediction[0] == 1 else 'Denied'}")

if __name__ == "__main__":
    main()

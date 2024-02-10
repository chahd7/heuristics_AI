"""/*=====Start Change Task 2=====*/"""
import eightpuzzle, search
import csv, random
import pandas as pd


#generate different puzzle scenarios

def get_scenarios(moves=3, n=300):
    scenarios = []  # Use a list instead of a set
    for i in range(n):
        puzzle = eightpuzzle.createRandomEightPuzzle(moves)
        elements = []
        for row in puzzle.cells:
            for cell in row:
                elements.append(cell)
        scenarios.append(tuple(elements))

    return scenarios




#save them into a scenarios csv file
def save_csv(scenarios, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for scenario in scenarios:
            writer.writerow(scenario)


#load from csv
def load_scenarios(filename='scenarios1.csv'):
    """Load 8-puzzle scenarios from a CSV file."""
    scenarios = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            scenario = list(map(int, row))
            scenarios.append(scenario)
    return scenarios


#get the info about each heuristic
def solve_and_record(problem, heuristic):
    """Solve the puzzle with the specified heuristic and record metrics."""
    path, expanded_nodes, fringe_size, depth = search.aStarSearch(problem, heuristic)
    return {'path': path, 'expanded_nodes': expanded_nodes, 'fringe_size': fringe_size, 'depth': depth}


def compare_heuristics(scenarios, heuristics, output_file='results1.csv'):
    """Compare heuristics by solving scenarios and recording the results, including the path."""
    results = []

    for index, scenario in enumerate(scenarios, start=1):
        state = eightpuzzle.EightPuzzleState(scenario)  # Assuming EightPuzzleState initialization
        problem = eightpuzzle.EightPuzzleSearchProblem(state)  # Assuming problem initialization
        
        for heuristic_name, heuristic in heuristics.items():
            path, expanded_nodes, fringe_size, depth = search.aStarSearch(problem, heuristic)
            result = {
                'scenario': f's{index}',
                'heuristic': heuristic_name,
                'expanded_nodes': expanded_nodes,
                'fringe_size': fringe_size,
                'depth': depth,
                'path': '->'.join(path) 
            }
            results.append(result)
    
    # Write results to CSV
    fieldnames = ['scenario', 'heuristic', 'expanded_nodes', 'fringe_size', 'depth', 'path']
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    # Generate scenarios
    scenarios = get_scenarios(10, 200)
    
    # Save scenarios to CSV file
    save_csv(scenarios, 'scenarios1.csv')

    scenarios = load_scenarios()
    heuristics = {'h1': search.h1, 'h2': search.h2, 'h3': search.h3, 'h4': search.h4}
    compare_heuristics(scenarios, heuristics)

    # Load your CSV file
    df = pd.read_csv('results1.csv')

    # Assuming the correct column name is 'heuristic' instead of 'heuristics'
    # Calculate the averages for each heuristic
    average_metrics = df.groupby('heuristic')[['expanded_nodes', 'fringe_size', 'depth']].mean()

    print(average_metrics)

if __name__ == "__main__":
    main()


"""/*=====End Change Task 2 =====*/"""
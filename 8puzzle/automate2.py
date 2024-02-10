import eightpuzzle, search
import csv, random
import pandas as pd

def get_scenarios(moves=3, n=200):
    scenarios = []
    for i in range(n):
        puzzle = eightpuzzle.createRandomEightPuzzle(moves)
        elements = []
        for row in puzzle.cells:
            for cell in row:
                elements.append(cell)
        scenarios.append(tuple(elements))
    return scenarios

def save_csv(scenarios, file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for scenario in scenarios:
            writer.writerow(scenario)
    print(f"Scenarios saved to {file_name}.")

def load_scenarios(filename='scenarios2.csv'):
    scenarios = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            scenario = list(map(int, row))
            scenarios.append(scenario)
    print(f"Loaded {len(scenarios)} scenarios from {filename}.")
    return scenarios

def run_search(problem, search_method, heuristic=None):
    try:
        heuristic_name = heuristic.__name__ if heuristic else 'None'
        print(f"Running search: {search_method.__name__}, heuristic: {heuristic_name}")
        
        if heuristic:
            result = search_method(problem, heuristic)
        else:
            result = search_method(problem)
        
        if len(result) == 4:  # Assuming result format is (path, expanded_nodes, fringe_size, depth)
            return result
        else:
            raise ValueError("Incorrect return format")

    except Exception as e:
        print(f"Error during search with {search_method.__name__}: {e}")
        return [], 0, 0, 0  # Default return format in case of an error




def compare_methods(scenarios, methods, output_file='results2.csv'):
    results = []
    print(f"Comparing search methods with {len(scenarios)} scenarios.")
    
    for index, scenario in enumerate(scenarios):
        state = eightpuzzle.EightPuzzleState(scenario)  # Adjust for your problem's state initialization
        problem = eightpuzzle.EightPuzzleSearchProblem(state)  # Adjust for your problem definition
        
        for method_name, method_details in methods.items():
            search_method = method_details['method']
            heuristic = method_details.get('heuristic')
            
            path, expanded_nodes, fringe_size, depth = run_search(problem, search_method, heuristic)
            
            # Convert the path to a string representation, if it's not already
            path_str = '->'.join(str(step) for step in path) if path else "No Path"
            
            results.append({
                'scenario': f"Scenario {index + 1}",
                'method': method_name,
                'expanded_nodes': expanded_nodes,
                'fringe_size': fringe_size,
                'depth': depth,
                'path': path_str  # Including path in the results
            })

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['scenario', 'method', 'expanded_nodes', 'fringe_size', 'depth', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results written to {output_file}.")


def main():
    scenarios = get_scenarios(10, 100)
    save_csv(scenarios, 'scenarios2.csv')
    scenarios = load_scenarios()
    
    methods = {
        'BFS': {'method': search.breadthFirstSearch},
        'DFS': {'method': search.depthFirstSearch},
        'UCS': {'method': search.uniformCostSearch},
        'A*': {'method': search.aStarSearch, 'heuristic': search.nullHeuristic},
        'h3': {'method': search.aStarSearch, 'heuristic': search.h3}
    }
    compare_methods(scenarios, methods)
    

   # Load your CSV file
    df = pd.read_csv('results2.csv')

    # Filter out DFS results


    # Calculate the averages for each method 
    average_metrics = df.groupby('method')[['expanded_nodes', 'fringe_size', 'depth']].mean()

    print(average_metrics)


if __name__ == "__main__":
    main()

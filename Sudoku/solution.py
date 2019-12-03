assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
	return [s+t for s in A for t in B]

boxes = cross(rows,cols)
row_units = [cross(r, cols) for r in rows]
col_units = [cross(rows, c) for c in cols]
diag_units = [[m+str(n) for m,n in zip(rows,cols)],[m+str(n) for m,n in zip(rows,cols[::-1])]]
square_units = [cross(rs,cs) for rs in (rows[:3],rows[3:6],rows[6:]) for cs in (cols[:3],cols[3:6],cols[6:])]
unitlist = row_units+col_units+diag_units+square_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(updated_values):
	"""
	The logic followed in this function is 
		1. loop through each unit in unitlist to find naked twins in that unit
		2. Once naked twins found then remove all naked twins digits from all other boxes in that unit
		3. Any update can possibly produce a naked twins in the same unit or other units
		4. Then break the loop and go to step 1 to find naked twins formed by this function
	"""
	updated = True
	"""While loop keep running as long as long as a box is updated"""
	while(updated):
		updated = False
		"""This loop covers logic of going through each unit in unitlist"""
		for u in unitlist:
			if updated is True:
				break				
			twins = {}
			unit_values = {k:updated_values[k] for k in u if len(updated_values[k]) == 2}		
			""" This loop groups all two digit boxes by box values, Sample shown below.
				sample-> this loop produces {'11':['A1','A7'],'12':['A2'],'56':['A8']}
			"""
			for k,v in sorted(unit_values.items()):
				twins.setdefault(v, []).append(k)
			
			twins = {v:kl for v,kl in twins.items() if len(kl)>1}
			"""This loop goes through dict of twins produced by above loop 
			Note: This can be improved by updating all peers instead of updating peers in unit"""
			for value,keylist in twins.items():
				for box in list(set(u) - set(keylist)):
					oldVal = updated_values[box]
					table = str.maketrans(dict.fromkeys(value))
					updatedVal = updated_values[box].translate(table)
					""" Blow two lines along with while loop covers new naked twins formed by the update"""
					if oldVal != updatedVal:
						updated = True
					
					updated_values = assign_value(updated_values,box,updatedVal)

	return updated_values
	
    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers

def grid_values(grid):
	value_list = [x.replace('.',cols) for x in grid]
	grid_dict = dict(zip(boxes,value_list))

	return grid_dict

def display(values):
	width = 1+max(len(values[k]) for k in boxes)
	line = '+'.join(['-'*(width*3)]*3)
	for r in rows:
		print(''.join(values[r+c].center(width)+('|' if c in '36' else '') for c in cols))
		if r in 'CF':
			print(line)
	return
	
def eliminate(values):
	for key,data in values.items():
		if len(data) == 1:
			for unit in (u for u in unitlist if key in u):
				for element in (e for e in unit if key!=e):
					values[element] = values[element].replace(data,'')
	return values
	
def only_choice(values):
	for unit in unitlist:
		for key in (k for k in unit if len(values[k])>1):
			unit_val = [values[k] for k in unit if k!=key]
			for data in list(values[key]):
				if not(any(data in v for v in unit_val)):
					values[key] = data
					break
	return values

def reduce_puzzle(values):
	stalled = False
	while not stalled:
		solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
		values = eliminate(values)
		values = only_choice(values)
		values = naked_twins(values)
		solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
		stalled = solved_values_before == solved_values_after
		if len([box for box in values.keys() if len(values[box]) == 0]):
			return False
	return values

def search(values):
	values = reduce_puzzle(values)
	if values:
    # Choose one of the unfilled squares with the fewest possibilities
		unSolVals = {ke:len(vl) for ke,vl in values.items() if len(vl)>1}
		if len(unSolVals) > 0:
			fewestValKey = min(unSolVals,key=unSolVals.get)
            
            # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
			for data in values[fewestValKey]:
				newValues = values.copy()
				newValues = assign_value(newValues,fewestValKey,data)
				newValues = search(newValues)
				if newValues:
					return newValues
		else:
			return values
	else:
		return False

def solve(grid):
	gridDict = grid_values(grid)	
	return search(gridDict)

if __name__ == '__main__':
	diag_sudoku_grid = '9.1....8.8.5.7..4.2.4....6...7......5..............83.3..6......9................'
#	diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
	display(solve(diag_sudoku_grid))
	
	try:
		from visualize import visualize_assignments
		visualize_assignments(assignments)
		
	except SystemExit:
		pass
	except:
		print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')

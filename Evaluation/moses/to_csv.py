import ast
import csv

a = open('results.txt', 'r')
i = 0
lines = a.read().splitlines()
print(lines)
cols = ['model']
valDict = []
while i < len(lines):
    key = lines[i]
    values = ''.join(lines[i+1:i+1+13])
    if (len(cols) == 1):
        for keys in ast.literal_eval(values).keys():
            cols.append(keys)
    print("KEY IS ", key)
    print(ast.literal_eval(values))
    valDict.append({'model' : key})
    dict1 = ast.literal_eval(values)
    for key,values in dict1.items():
        print(float(values))
        dict1[key] = round(float(values), 3)
    valDict[-1].update(dict1)
    i+=14

with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = cols
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in valDict:
        writer.writerow(i)
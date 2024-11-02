import shutil

# Source file name
source_file = 'main/task10000.py'

# Number of copies to make
num_copies = 69

# Loop to create each copy
for i in range(1, num_copies + 1):
    # Generate the new file name
    new_file = f'main/task{10000 + i}.py'
    
    # Copy the file
    shutil.copy(source_file, new_file)

print(f'{num_copies} copies of {source_file} have been created.')

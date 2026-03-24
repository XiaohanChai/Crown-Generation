import os

def rename_files():
    directory = '/root/octfusion/data/mask_crown_350/closed_crown/lower'
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    files = os.listdir(directory)
    files.sort() # Ensure consistent order

    count = 0
    for filename in files:
        if filename.startswith('lower') and filename.endswith('.obj'):
            # Extract the number part
            # Assuming format lowerXXXX.obj
            try:
                number_part = filename[5:-4] # remove 'lower' and '.obj'
                # Verify it is a number
                int(number_part) 
                
                new_filename = f"data{number_part}.obj"
                
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                count += 1
            except ValueError:
                print(f"Skipping file with unexpected format: {filename}")
                continue
    
    print(f"Total files renamed: {count}")

if __name__ == "__main__":
    rename_files()

import os
import json

def convert_json_to_text(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_dir, output_filename)

            # Read JSON file
            with open(input_path, 'r') as json_file:
                data = json.load(json_file)

            # Write content to text file
            with open(output_path, 'w') as text_file:
                text_file.write(json.dumps(data, indent=2))

            print(f"Converted {filename} to {output_filename}")

# Set the input and output directories
input_directory = r"D:\SignLanguage\NOTEBOOKS\YOLO_DS_split\labels\val"
output_directory = r"D:\SignLanguage\NOTEBOOKS\YOLO_DS_split\labels\v_text"

# Run the conversion
convert_json_to_text(input_directory, output_directory)

print("Conversion complete!")
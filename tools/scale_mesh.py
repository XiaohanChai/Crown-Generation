import os
import numpy as np
import argparse

def load_vertices(path):
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Handle cases where there might be extra spaces
                # Filter out empty strings if multiple spaces exist
                parts = [p for p in parts if p]
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def get_scale(vertices):
    if len(vertices) == 0:
        return 1.0
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bbox_size = max_coords - min_coords
    # Use diagonal length as scale metric
    scale = np.linalg.norm(bbox_size)
    return scale

def scale_and_save(src_path, ref_path, output_path):
    print(f"Loading source: {src_path}")
    src_vertices = load_vertices(src_path)
    print(f"Loading reference: {ref_path}")
    ref_vertices = load_vertices(ref_path)

    src_scale = get_scale(src_vertices)
    ref_scale = get_scale(ref_vertices)

    if src_scale == 0:
        print("Error: Source object has 0 scale.")
        return

    scale_factor = ref_scale / src_scale
    print(f"Source scale (diagonal): {src_scale:.6f}")
    print(f"Reference scale (diagonal): {ref_scale:.6f}")
    print(f"Scaling factor: {scale_factor:.6f}")
    
    print(f"Writing scaled object to: {output_path}")
    with open(src_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('v '):
                parts = line.strip().split()
                parts = [p for p in parts if p]
                
                x = float(parts[1]) * scale_factor
                y = float(parts[2]) * scale_factor
                z = float(parts[3]) * scale_factor
                f_out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                f_out.write(line)
    print("Done.")

if __name__ == "__main__":
    # Default paths based on user request
    base_dir = "/root/octfusion/data/mask_crown_350/experiment/test_ryi"
    src_obj = os.path.join(base_dir, "generate.obj")
    ref_obj = os.path.join(base_dir, "origin.obj")
    out_obj = os.path.join(base_dir, "scaled.obj")

    parser = argparse.ArgumentParser(description="Scale generate.obj to match origin.obj size")
    parser.add_argument("--src", default=src_obj, help="Path to source obj file")
    parser.add_argument("--ref", default=ref_obj, help="Path to reference obj file")
    parser.add_argument("--out", default=out_obj, help="Path to output obj file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.src):
        print(f"Error: Source file not found at {args.src}")
    elif not os.path.exists(args.ref):
        print(f"Error: Reference file not found at {args.ref}")
    else:
        scale_and_save(args.src, args.ref, args.out)

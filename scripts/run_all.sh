# python tools/repair_mesh_mask.py --run run_mesh2sdf_split_mp
# python tools/repair_mesh_mask.py --run generate_dataset
# sh scripts/run_snet_vae.sh train vae crown_585
sh scripts/run_snet_uncond_mask_crown_lr.sh train lr crown_585
sh scripts/run_snet_uncond_mask_crown_hr.sh train hr crown_585
load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dma.ent", occ_335_c1_p0_s0.8
hide everything, occ_335_c1_p0_s0.8
show cartoon, occ_335_c1_p0_s0.8 and chain A+D
color palegreen, occ_335_c1_p0_s0.8 and chain A
color lightblue, occ_335_c1_p0_s0.8 and chain D
select hotspot_source, occ_335_c1_p0_s0.8 and ((chain A and resi 500))
select hotspot_target, occ_335_c1_p0_s0.8 and ((chain D and resi 41))
select hotspot_all, occ_335_c1_p0_s0.8 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_335_c1_p0_s0.8 and chain A+D
set_name hotspot_all, hotspot_occurrence_335
set_name hotspot_source, hotspot_source_335
set_name hotspot_target, hotspot_target_335
bg_color white
# patternId=0 support=0.8 graphId=393

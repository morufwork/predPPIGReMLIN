load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7bh9.ent", occ_137_c0_p0_s0.9
hide everything, occ_137_c0_p0_s0.9
show cartoon, occ_137_c0_p0_s0.9 and chain A+E
color palegreen, occ_137_c0_p0_s0.9 and chain A
color lightblue, occ_137_c0_p0_s0.9 and chain E
select hotspot_source, occ_137_c0_p0_s0.9 and ((chain A and resi 353))
select hotspot_target, occ_137_c0_p0_s0.9 and ((chain E and resi 501))
select hotspot_all, occ_137_c0_p0_s0.9 and ((chain A and resi 353) or (chain E and resi 501))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_137_c0_p0_s0.9 and chain A+E
set_name hotspot_all, hotspot_occurrence_137
set_name hotspot_source, hotspot_source_137
set_name hotspot_target, hotspot_target_137
bg_color white
# patternId=0 support=0.9 graphId=31

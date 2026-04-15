load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_866_c2_p0_s0.9
hide everything, occ_866_c2_p0_s0.9
show cartoon, occ_866_c2_p0_s0.9 and chain A+C
color palegreen, occ_866_c2_p0_s0.9 and chain A
color lightblue, occ_866_c2_p0_s0.9 and chain C
select hotspot_source, occ_866_c2_p0_s0.9 and ((chain A and resi 484))
select hotspot_target, occ_866_c2_p0_s0.9 and ((chain C and resi 67))
select hotspot_all, occ_866_c2_p0_s0.9 and ((chain A and resi 484) or (chain C and resi 67))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_866_c2_p0_s0.9 and chain A+C
set_name hotspot_all, hotspot_occurrence_866
set_name hotspot_source, hotspot_source_866
set_name hotspot_target, hotspot_target_866
bg_color white
# patternId=0 support=0.9 graphId=186

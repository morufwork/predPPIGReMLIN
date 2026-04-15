load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_656_c2_p0_s0.8
hide everything, occ_656_c2_p0_s0.8
show cartoon, occ_656_c2_p0_s0.8 and chain A+C
color palegreen, occ_656_c2_p0_s0.8 and chain A
color lightblue, occ_656_c2_p0_s0.8 and chain C
select hotspot_source, occ_656_c2_p0_s0.8 and ((chain A and resi 484))
select hotspot_target, occ_656_c2_p0_s0.8 and ((chain C and resi 67))
select hotspot_all, occ_656_c2_p0_s0.8 and ((chain A and resi 484) or (chain C and resi 67))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_656_c2_p0_s0.8 and chain A+C
set_name hotspot_all, hotspot_occurrence_656
set_name hotspot_source, hotspot_source_656
set_name hotspot_target, hotspot_target_656
bg_color white
# patternId=0 support=0.8 graphId=186

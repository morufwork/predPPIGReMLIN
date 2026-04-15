load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7pki.ent", occ_317_c1_p0_s0.8
hide everything, occ_317_c1_p0_s0.8
show cartoon, occ_317_c1_p0_s0.8 and chain A+E
color palegreen, occ_317_c1_p0_s0.8 and chain A
color lightblue, occ_317_c1_p0_s0.8 and chain E
select hotspot_source, occ_317_c1_p0_s0.8 and ((chain A and resi 41))
select hotspot_target, occ_317_c1_p0_s0.8 and ((chain E and resi 496))
select hotspot_all, occ_317_c1_p0_s0.8 and ((chain A and resi 41) or (chain E and resi 496))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_317_c1_p0_s0.8 and chain A+E
set_name hotspot_all, hotspot_occurrence_317
set_name hotspot_source, hotspot_source_317
set_name hotspot_target, hotspot_target_317
bg_color white
# patternId=0 support=0.8 graphId=174

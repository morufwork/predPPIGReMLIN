load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6m0j.ent", occ_2732_c5_p0_s0.8
hide everything, occ_2732_c5_p0_s0.8
show cartoon, occ_2732_c5_p0_s0.8 and chain A+E
color palegreen, occ_2732_c5_p0_s0.8 and chain A
color lightblue, occ_2732_c5_p0_s0.8 and chain E
select hotspot_source, occ_2732_c5_p0_s0.8 and ((chain A and resi 82) or (chain A and resi 83))
select hotspot_target, occ_2732_c5_p0_s0.8 and ((chain E and resi 486))
select hotspot_all, occ_2732_c5_p0_s0.8 and ((chain A and resi 82) or (chain A and resi 83) or (chain E and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2732_c5_p0_s0.8 and chain A+E
set_name hotspot_all, hotspot_occurrence_2732
set_name hotspot_source, hotspot_source_2732
set_name hotspot_target, hotspot_target_2732
bg_color white
# patternId=0 support=0.8 graphId=21

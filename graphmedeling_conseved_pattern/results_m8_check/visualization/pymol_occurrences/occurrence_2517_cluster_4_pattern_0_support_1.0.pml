load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7pki.ent", occ_2517_c4_p0_s1.0
hide everything, occ_2517_c4_p0_s1.0
show cartoon, occ_2517_c4_p0_s1.0 and chain A+E
color palegreen, occ_2517_c4_p0_s1.0 and chain A
color lightblue, occ_2517_c4_p0_s1.0 and chain E
select hotspot_source, occ_2517_c4_p0_s1.0 and ((chain A and resi 27))
select hotspot_target, occ_2517_c4_p0_s1.0 and ((chain E and resi 452) or (chain E and resi 485))
select hotspot_all, occ_2517_c4_p0_s1.0 and ((chain A and resi 27) or (chain E and resi 452) or (chain E and resi 485))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2517_c4_p0_s1.0 and chain A+E
set_name hotspot_all, hotspot_occurrence_2517
set_name hotspot_source, hotspot_source_2517
set_name hotspot_target, hotspot_target_2517
bg_color white
# patternId=0 support=1.0 graphId=168

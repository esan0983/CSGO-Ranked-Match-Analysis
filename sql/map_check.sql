-- Load the CSV's depending on what software you use.
SELECT * FROM csgo
LEFT JOIN maps ON csgo.map == maps.map
WHERE att_pos_x >= StartX 
    AND att_pos_x <= EndX 
    AND att_pos_y >= StartY 
    AND att_pos_y <= EndY
    AND vic_pos_x >= StartX 
    AND vic_pos_x <= EndX 
    AND vic_pos_y >= StartY 
    AND vic_pos_y <= EndY;
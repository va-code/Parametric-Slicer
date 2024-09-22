function random_shape() = floor(rands(0, 2, 1)[0]);

module fractal(depth, size) {
        union(){
        if (depth > 0) {
            //echo("Depth:")
            //echo(depth)
            for (x = [-1, 1], y = [-1, 1], z = [-1, 1]) {
                translate([x*size/5, y*size/5, z*size/5])
                    fractal(depth - 1, size / 2);
            }
        } else {
            shape = random_shape();
            //echo("shape:")
            //echo(shape)
            if (shape == 0) {
                cube([size, size, size], center=true);
            } else if (shape == 1) {
                sphere(size / 2,$fn=100);
            } else if (shape == 2) {
                cylinder(h = size, r = size / 2, center=true);
            }
        }}
}

fractal(2, 5);
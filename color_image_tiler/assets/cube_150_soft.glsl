#version 150 core
out vec4 o_Color;

uniform b_Globals {
    float u_time;
};

uniform sampler2D t_point;
uniform sampler2D t_color;

void main() {
    float grid_size = PARTICLE_FREEDOM / float(TILE_COUNT_SQRT);

    float closest_d = 1;
    int closest_i = 0;
    float second_closest_d = 1;
    int second_closest_i = 0;

    float input_x = gl_FragCoord.x / 640;
    float input_y = 1 - gl_FragCoord.y / 640;

    int sample_grid_x = int(floor(input_x * TILE_COUNT_SQRT));
    int sample_grid_y = int(floor(input_y * TILE_COUNT_SQRT));

    float inner_coord_x = input_x * TILE_COUNT_SQRT - sample_grid_x;
    float inner_coord_y = input_y * TILE_COUNT_SQRT - sample_grid_y;

    if (inner_coord_x < 0.03 ||
            inner_coord_y < 0.03) {
        o_Color = vec4(1, 0, 1, 0);
        return;
    }

    for (int cell_dx = -PARTICLE_FREEDOM; cell_dx <= PARTICLE_FREEDOM; cell_dx++) {
        for (int cell_dy = -PARTICLE_FREEDOM; cell_dy <= PARTICLE_FREEDOM; cell_dy++) {
            int cell_x = sample_grid_x + cell_dx;
            int cell_y = sample_grid_y + cell_dy;

            if (cell_x < 0 || cell_y < 0 || cell_x >= TILE_COUNT_SQRT || cell_y >= TILE_COUNT_SQRT) {
                continue;
            }

            int i = (cell_x * TILE_COUNT_SQRT) + cell_y;

            vec4 relative = texture(t_point, vec2((i + .5) / TILE_COUNT, 0));

            float seed_x = relative.x * grid_size + float(cell_x) / TILE_COUNT_SQRT;
            float seed_y = relative.y * grid_size + float(cell_y) / TILE_COUNT_SQRT;

            float x_d = seed_x - input_x;
            float y_d = seed_y - input_y;

            float d_2 = (x_d * x_d) + (y_d * y_d);

            if (sqrt(d_2) < .002) {
                o_Color = vec4(1, 0, 1, 1);
                return;
            }

            if (closest_d > d_2) {
                second_closest_d = closest_d;
                second_closest_i = closest_i;

                closest_d = d_2;
                closest_i = i;
            } else if (second_closest_d > d_2) {
                second_closest_d = d_2;
                second_closest_i = i;
            }
        }
    }

    vec4 closest_col = vec4(texture(t_color, vec2((closest_i + .5) / TILE_COUNT, 0)).xyz, 1);
    vec4 second_closest_col = vec4(texture(t_color, vec2((second_closest_i + .5) / TILE_COUNT, 0)).xyz, 1);

    float factor = closest_d / second_closest_d;

    if (factor < TILE_BIAS) {
        o_Color = closest_col;
    } else {
        factor = (factor - TILE_BIAS) / (1 - TILE_BIAS);
        o_Color = mix(closest_col, mix(closest_col, second_closest_col, .5), factor);
    }

    // o_Color = vec4(factor,0,0,1);
    //  o_Color = second_closest_col;
}

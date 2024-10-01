#version 150 core
out vec4 o_Color;

uniform b_Globals {
    float u_time;
};

uniform sampler2D t_point;
uniform sampler2D t_color;

vec2 projectPointToVector(vec2 P, vec2 V) {
    // Normalize the vector V
    vec2 normalizedV = normalize(V);

    // Calculate the projection length of P onto V
    float projectionLength = dot(P, normalizedV);

    // Calculate the projected point
    vec2 projectedPoint = projectionLength * normalizedV;

    return projectedPoint;
}

void main() {
    float grid_size = PARTICLE_FREEDOM / float(TILE_COUNT_SQRT);

    float closest_d = 1;
    int closest_i = 0;
    vec2 closest_point = vec2(0, 0);
    float second_closest_d = 1;
    int second_closest_i = 0;
    vec2 second_closest_point = vec2(0, 0);

    float input_x = gl_FragCoord.x / 640;
    float input_y = 1 - gl_FragCoord.y / 640;
    vec2 input_point = vec2(input_x, input_y);

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

            vec2 seed = vec2(relative.x * grid_size + float(cell_x) / TILE_COUNT_SQRT, relative.y * grid_size + float(cell_y) / TILE_COUNT_SQRT);

            float d = length(seed - input_point);

            if (d < .002) {
                o_Color = vec4(1, 0, 1, 1);
                return;
            }

            if (closest_d > d) {
                second_closest_d = closest_d;
                second_closest_i = closest_i;
                second_closest_point = closest_point;

                closest_d = d;
                closest_i = i;
                closest_point = seed;
            } else if (second_closest_d > d) {
                second_closest_d = d;
                second_closest_i = i;
                second_closest_point = seed;
            }
        }
    }

    vec4 closest_col = vec4(texture(t_color, vec2((closest_i + .5) / TILE_COUNT, 0)).xyz, 1);
    vec4 second_closest_col = vec4(texture(t_color, vec2((second_closest_i + .5) / TILE_COUNT, 0)).xyz, 1);

    vec2 gradient_direction = normalize(closest_point - second_closest_point);

    vec2 projected_closest_p = projectPointToVector(closest_point, gradient_direction);
    vec2 projected_second_closest_p = projectPointToVector(second_closest_point, gradient_direction);
    vec2 projected_input = projectPointToVector(input_point, gradient_direction);

    float projected_closest_d = length(projected_closest_p - projected_input);
    float projected_second_closest_d = length(projected_second_closest_p - projected_input);

    float factor = projected_closest_d / projected_second_closest_d;

    if (factor < TILE_BIAS) {
        o_Color = closest_col;
    } else {
        factor = (factor - TILE_BIAS) / (1 - TILE_BIAS);
        o_Color = mix(closest_col second_closest_col, factor / 2);
    }

    // o_Color = vec4(factor,0,0,1);
    //  o_Color = second_closest_col;
}

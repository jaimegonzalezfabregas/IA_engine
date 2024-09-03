#version 150 core
out vec4 o_Color;

uniform b_Globals {
    float u_time;
};

uniform sampler2D t_point;
uniform sampler2D t_color;

void main() {

    float total_inverse_d = 0;

    for(int i=0;i<TILE_COUNT;++i)
    {
        vec4 color = texture(t_color, vec2((i + 0.5)/TILE_COUNT,0));
        vec4 point = texture(t_point, vec2((i + 0.5)/TILE_COUNT,0));

        float x_d = point.x - gl_FragCoord.x / 640;
        float y_d = point.y - gl_FragCoord.y / 640;

        float d_2 =  (x_d * x_d) + (y_d * y_d);

        float d_4 = d_2 * d_2;
        float d_8 = d_4 * d_4;
        float d_16 = d_8 * d_8;

        total_inverse_d += 1 / d_16;
    }

    vec3 ret = vec3(0);


    for(int i=0;i<TILE_COUNT;++i)
    {
        vec4 color = texture(t_color, vec2((i + 0.5)/TILE_COUNT,0));
        vec4 point = texture(t_point, vec2((i + 0.5)/TILE_COUNT,0));

        float x_d = point.x - gl_FragCoord.x / 640;
        float y_d = point.y - gl_FragCoord.y / 640;

        float d_2 = + (x_d * x_d) + (y_d * y_d);

        float d_4 = d_2 * d_2;
        float d_8 = d_4 * d_4;
        float d_16 = d_8 * d_8;

        ret += color.xyz / d_16 / total_inverse_d;

    }



    o_Color = vec4(ret, 1);

}
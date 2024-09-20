#version 150 core
out vec4 o_Color;

uniform b_Globals {
    float u_time;
};

uniform sampler2D t_point;
uniform sampler2D t_color;

void main() {

    float closest_d = 1;
    int closest_i = 0;

    for(int i=0;i<TILE_COUNT;++i)
    {
        vec4 color = texture(t_color, vec2((i + 0.5)/TILE_COUNT,0));
        vec4 point = texture(t_point, vec2((i + 0.5)/TILE_COUNT,0));

        float x_d = point.x - gl_FragCoord.x / 640;
        float y_d = point.y - gl_FragCoord.y / 640;

        float d_2 =  (x_d * x_d) + (y_d * y_d);

        if(closest_d > d_2){
            closest_d = d_2;
            closest_i = i;
        }
    }

    o_Color = vec4(texture(t_color, vec2((closest_i + 0.5)/TILE_COUNT,0)).xyz, 1);

}
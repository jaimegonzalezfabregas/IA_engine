#version 150 core
out vec4 o_Color;

uniform b_Globals {
    float u_time;
};

uniform sampler2D t_point;
uniform sampler2D t_color;

void main() {

    float BIAS = 0.9;

    float closest_d = 1;
    int closest_i = 0;
    float second_closest_d = 1;
    int second_closest_i = 0;

    for(int i=0;i<TILE_COUNT;++i)
    {
        vec4 point = texture(t_point, vec2((i + 0.5)/TILE_COUNT,0));

        float x_d = point.x - gl_FragCoord.x / 640;
        float y_d = point.y - gl_FragCoord.y / 640;

        float d_2 =  (x_d * x_d) + (y_d * y_d);

        if(closest_d > d_2){
            
            second_closest_d = closest_d;
            second_closest_i = closest_i;
            
            closest_d = d_2;
            closest_i = i;
        }else if(second_closest_d > d_2){
            second_closest_d = d_2;
            second_closest_i = i;
        }
    }

  

    vec4 closest_col = vec4(texture(t_color, vec2((closest_i + 0.5)/TILE_COUNT,0)).xyz, 1);   
    vec4 second_closest_col = vec4(texture(t_color, vec2((second_closest_i + 0.5)/TILE_COUNT,0)).xyz, 1);

    float factor = closest_d / second_closest_d;

    if(factor < BIAS){
        factor = 0;
    }else{
        factor = (factor -BIAS) / (1 - BIAS);
    }

    o_Color = mix(closest_col, mix(closest_col, second_closest_col, 0.5), factor);
    // o_Color = vec4(factor,0,0,1);
    //  o_Color = second_closest_col;

    

}
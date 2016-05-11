#version 330
layout (location = 1) in vec3 Vertex_position;
layout (location = 2) in vec3 Vertex_normal;
layout (location = 3) in vec2 Texture_position;

out vec2 texture_coordinates;
out vec3 vertex_to_camera;
out vec3 normal_to_camera;

uniform mat4 view_matrix;
uniform mat4 perspective_matrix;


void main()
{
    normal_to_camera = vec3(view_matrix * vec4(Vertex_normal, 0.0));
    vertex_to_camera = vec3(view_matrix * vec4(Vertex_position, 1.0));

    texture_coordinates =  Texture_position ;
    gl_Position = perspective_matrix * view_matrix * vec4(Vertex_position, 1.0);
}
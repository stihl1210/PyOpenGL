in vec2 texture_coordinates;
in vec3 vertex_to_camera;
in vec3 normal_to_camera;

out vec4 colorOut;

uniform mat4 view_matrix;

uniform vec4 Global_ambient;
uniform vec4 Light_ambient;
uniform vec4 Light_diffuse;
uniform vec3 Light_location;
uniform vec4 Light_specular;

uniform vec4 Material_ambient;
uniform vec4 Material_diffuse;
uniform vec4 Material_specular;
uniform sampler2D texture;

uniform float shininess;


void main() {

   // Ambient
   vec4 ambient_intense = Light_ambient * Material_ambient;

   // Diffuse
   vec3 distance_to_light = vec3(view_matrix * vec4(Light_location, 1.0)) - vertex_to_camera;
   vec3 light_direction = normalize(distance_to_light);
   float dot_product = dot(light_direction, normal_to_camera);
   dot_product = max(dot_product, 0.0);
   vec4 diffuse_intense = Light_diffuse * Material_diffuse * dot_product;

   // Specular
   vec3 reflection = reflect(-light_direction, normal_to_camera);
   vec3 surface_to_camera = normalize(-vertex_to_camera);
   float dot_specular = dot(reflection, surface_to_camera);
   dot_specular = max(dot_specular, 0.0);
   float specular_power = 10;
   float specular_factor = pow(dot_specular, specular_power);
   vec4 specular_intense = Light_specular * Material_specular * specular_factor;

   vec4 texel = texture(texture, texture_coordinates);

   vec3 fog_colour = vec3(0.5, 0.5, 0.5);
   float min_fog_distance = 5.0;
   float max_fog_distance = 1000.0;
   float distance = length(-vertex_to_camera);
   float fog_factor = (distance - min_fog_distance) / (max_fog_distance - min_fog_distance);
   fog_factor = clamp(fog_factor, 0.0, 1.0);

   vec4 light_and_texture = texel ;

   colorOut.rgb = mix(light_and_texture.rgb, fog_colour, fog_factor);
}
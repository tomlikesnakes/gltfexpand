#version 330

// Input attributes from vertex shader
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec3 fragPosition;

// Output fragment color
out vec4 finalColor;

// Uniforms
uniform sampler2D texture0;
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform float ambientStrength;

void main() {
    // Texture color
    vec4 texColor = texture(texture0, fragTexCoord) * fragColor;
    
    // Ambient light
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse light
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular light
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Combine results
    vec3 result = (ambient + diffuse + specular) * texColor.rgb;
    finalColor = vec4(result, texColor.a);
}

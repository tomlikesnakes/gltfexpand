// gltf_loader.c

// Include necessary headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <raylib.h>
#include "raymath.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// Define maximum limits
#define MAX_BONES 256
#define MAX_ANIMATIONS 64
#define MAX_MESHES 64
#define MAX_NODES 512
#define MAX_MATERIALS 64
#define MAX_TEXTURES 64

// Data structures
typedef struct {
	Matrix transform;
	Matrix globalTransform;
	int parentIndex;
	int childCount;
	int childIndices[MAX_NODES];
	const char* name;
	bool hasAnimation;
} Node;

typedef struct {
	int nodeIndex;
	const char* name;
} MyBoneInfo;

typedef struct {
	int boneCount;
	MyBoneInfo bones[MAX_BONES];
} Skeleton;

typedef struct {
	int keyframeCount;
	float* times;
	Vector3* translations;
	Quaternion* rotations;
	Vector3* scales;
} AnimationChannel;

typedef struct {
	const char* name;
	int channelCount;
	AnimationChannel channels[MAX_NODES];
	float duration;
} AnimationData;

typedef struct {
	int animationCount;
	AnimationData animations[MAX_ANIMATIONS];
	int currentAnimation;
	bool isPlaying;
	float time;
} AnimationState;

typedef struct {
	Texture2D textures[MAX_TEXTURES];
	int textureCount;
} TextureCache;

// Function prototypes
cgltf_data* LoadGLTFFile(const char* filename);
void FreeGLTFData(cgltf_data* data);
void LoadNodes(cgltf_data* data, Node* nodes, int* nodeCount);
void BuildNodeHierarchy(Node* nodes, int nodeCount, cgltf_data* data);
void LoadSkeleton(cgltf_data* data, Skeleton* skeleton, Node* nodes, int nodeCount);
void LoadAnimations(cgltf_data* data, AnimationState* animationState, Node* nodes, int nodeCount);
Model LoadModelFromGLTF(cgltf_data* data, Skeleton* skeleton, TextureCache* textureCache, Node* nodes, int nodeCount, MyBoneInfo** outMyBoneInfos, Matrix** outInverseBindPose);
void UpdateAnimation(AnimationState* animationState, Node* nodes, int nodeCount);
void ApplyNodeTransforms(Node* nodes, int nodeCount, Skeleton* skeleton, MyBoneInfo* myBoneInfos, Matrix* inverseBindPose, Matrix* boneTransforms);
void UpdateGlobalTransforms(Node* nodes, int nodeIndex);
void FreeAnimationData(AnimationState* animationState, int nodeCount);
void LoadGLTFMaterials(cgltf_data* data, Model* model, TextureCache* textureCache);
Texture2D LoadTextureFromImageUri(const char* basePath, const char* uri);
void PrintNodeHierarchy(Node* nodes, int nodeIndex, int depth);
void PrintBoneMatrices(Matrix* boneTransforms, Skeleton* skeleton);
void PrintDebugInfo(AnimationState* animationState, Skeleton* skeleton, Node* nodes, int nodeCount, Matrix* boneTransforms);
void PrintMeshData(Mesh mesh);
void SetShaderValueMatrixArray(Shader shader, int locIndex, const Matrix* matrices, int count);
Vector3 MatrixToTranslation(Matrix m);
Vector3 MatrixToScale(Matrix m);
Quaternion QuaternionFromMatrix(Matrix m);
void DrawSkeleton(Model model, Matrix* boneTransforms);
Matrix TransformToMatrix(Transform t);
Matrix MatrixFromFloatArray(float* mat);

const char *basicVertexShader =
	"#version 330\n"
	"in vec3 vertexPosition;\n"
	"in vec3 vertexNormal;\n"
	"uniform mat4 mvp;\n"
	"out vec3 fragNormal;\n"
	"void main()\n"
	"{\n"
	"    fragNormal = vertexNormal;\n"
	"    gl_Position = mvp * vec4(vertexPosition, 1.0);\n"
	"}\n";

const char *basicFragmentShader =
	"#version 330\n"
	"in vec3 fragNormal;\n"
	"out vec4 finalColor;\n"
	"uniform vec4 color;\n"
	"void main()\n"
	"{\n"
	"    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));\n"
	"    float intensity = max(dot(fragNormal, lightDir), 0.0);\n"
	"    finalColor = color * intensity;\n"
	"}\n";

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_gltf_file>\n", argv[0]);
        return -1;
    }

    // Initialize Raylib
    SetConfigFlags(FLAG_MSAA_4X_HINT);  // Enable MSAA for smoother rendering
    InitWindow(1280, 720, "Custom glTF Loader with Animations");
    SetTargetFPS(60);


    // Load glTF data
    cgltf_data* data = LoadGLTFFile(argv[1]);
    if (!data) {
        printf("Failed to load glTF file.\n");
        CloseWindow();
        return -1;
    }

    // Load nodes
    Node nodes[MAX_NODES] = {0};
    int nodeCount = 0;
    LoadNodes(data, nodes, &nodeCount);
    BuildNodeHierarchy(nodes, nodeCount, data);

    // Print node hierarchy for debugging
    printf("\n--- Node Hierarchy ---\n");
    for (int i = 0; i < nodeCount; ++i) {
        if (nodes[i].parentIndex == -1) {
            PrintNodeHierarchy(nodes, i, 0);
        }
    }

    // Load skeleton
    Skeleton skeleton = {0};
    LoadSkeleton(data, &skeleton, nodes, nodeCount);

    // Load animations
    AnimationState animationState = {0};
    LoadAnimations(data, &animationState, nodes, nodeCount);
    animationState.isPlaying = true;

    // Load model
    TextureCache textureCache = {0};
    MyBoneInfo* myBoneInfos = NULL;
    Matrix* inverseBindPose = NULL;
    Model model = LoadModelFromGLTF(data, &skeleton, &textureCache, nodes, nodeCount, &myBoneInfos, &inverseBindPose);

    // Create boneTransforms array
    Matrix* boneTransforms = (Matrix*)malloc(sizeof(Matrix) * skeleton.boneCount);
    if (!boneTransforms) {
        printf("Failed to allocate memory for bone transforms.\n");
        CloseWindow();
        return -1;
    }
    for (int i = 0; i < skeleton.boneCount; ++i) {
        boneTransforms[i] = MatrixIdentity();
    }

    // Camera setup
    Camera camera = {0};
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // Print mesh and material information
    printf("\n--- Mesh Information ---\n");
    for (int i = 0; i < model.meshCount; i++) {
        printf("Mesh %d:\n", i);
        PrintMeshData(model.meshes[i]);
        printf("\n");
    }

    printf("\n--- Material Information ---\n");
    for (int i = 0; i < model.materialCount; i++) {
        printf("Material %d:\n", i);
        printf("  Shader ID: %u\n", model.materials[i].shader.id);
        printf("  Color: R:%d G:%d B:%d A:%d\n",
            model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color.r,
            model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color.g,
            model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color.b,
            model.materials[i].maps[MATERIAL_MAP_DIFFUSE].color.a);
    }

    // Load and set up shader
    Shader basicShader = LoadShaderFromMemory(basicVertexShader, basicFragmentShader);
    int mvpLoc = GetShaderLocation(basicShader, "mvp");
    int colorLoc = GetShaderLocation(basicShader, "color");
    int boneTransformsLoc = GetShaderLocation(basicShader, "boneTransforms");

    // Set a default color
    Vector4 defaultColor = {1.0f, 1.0f, 1.0f, 1.0f};
    SetShaderValue(basicShader, colorLoc, &defaultColor, SHADER_UNIFORM_VEC4);

    if (basicShader.id == 0) {
        printf("Failed to load shader\n");
        CloseWindow();
        return -1;
    } else {
        printf("Shader loaded successfully, ID: %u\n", basicShader.id);
        printf("Shader locations: mvp=%d, color=%d, boneTransforms=%d\n", mvpLoc, colorLoc, boneTransformsLoc);
    }

    // Assign shader to model materials
    for (int i = 0; i < model.materialCount; ++i) {
        model.materials[i].shader = basicShader;
    }

    // UI variables
    bool showDebugInfo = false;
    bool showSkeleton = true;
    float modelScale = 1.0f;
    Vector3 modelPosition = {0.0f, 0.0f, 0.0f};

    printf("Starting main loop. Press ESC to exit.\n");

    while (!WindowShouldClose()) {
        // Input handling
        if (IsKeyPressed(KEY_SPACE)) {
            animationState.isPlaying = !animationState.isPlaying;
            printf("Animation %s\n", animationState.isPlaying ? "resumed" : "paused");
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            animationState.currentAnimation = (animationState.currentAnimation + 1) % animationState.animationCount;
            animationState.time = 0.0f;
            printf("Switched to animation: %s\n", animationState.animations[animationState.currentAnimation].name);
        }
        if (IsKeyPressed(KEY_LEFT)) {
            animationState.currentAnimation = (animationState.currentAnimation - 1 + animationState.animationCount) % animationState.animationCount;
            animationState.time = 0.0f;
            printf("Switched to animation: %s\n", animationState.animations[animationState.currentAnimation].name);
        }
        if (IsKeyPressed(KEY_D)) showDebugInfo = !showDebugInfo;
        if (IsKeyPressed(KEY_S)) showSkeleton = !showSkeleton;
        
        // Model scaling
        if (IsKeyDown(KEY_KP_ADD)) modelScale *= 1.01f;
        if (IsKeyDown(KEY_KP_SUBTRACT)) modelScale *= 0.99f;
        
        // Model position
        if (IsKeyDown(KEY_W)) modelPosition.z -= 0.1f;
        if (IsKeyDown(KEY_S)) modelPosition.z += 0.1f;
        if (IsKeyDown(KEY_A)) modelPosition.x -= 0.1f;
        if (IsKeyDown(KEY_D)) modelPosition.x += 0.1f;
        if (IsKeyDown(KEY_Q)) modelPosition.y += 0.1f;
        if (IsKeyDown(KEY_E)) modelPosition.y -= 0.1f;

        UpdateCamera(&camera, CAMERA_ORBITAL);

        if (animationState.isPlaying && animationState.animationCount > 0) {
            animationState.time += GetFrameTime();
            AnimationData* currentAnimation = &animationState.animations[animationState.currentAnimation];
            if (animationState.time > currentAnimation->duration) {
                animationState.time = fmodf(animationState.time, currentAnimation->duration);
            }
        }

        UpdateAnimation(&animationState, nodes, nodeCount);
        ApplyNodeTransforms(nodes, nodeCount, &skeleton, myBoneInfos, inverseBindPose, boneTransforms);

        int boneCount = (skeleton.boneCount > MAX_BONES) ? MAX_BONES : skeleton.boneCount;
        SetShaderValueMatrixArray(basicShader, boneTransformsLoc, boneTransforms, boneCount);

        Matrix projection = MatrixFrustum(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 1000.0f);
        Matrix view = GetCameraMatrix(camera);
        Matrix modelMatrix = MatrixMultiply(MatrixTranslate(modelPosition.x, modelPosition.y, modelPosition.z),
                                            MatrixScale(modelScale, modelScale, modelScale));
        Matrix mvp = MatrixMultiply(MatrixMultiply(modelMatrix, view), projection);

        BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(camera);
                DrawGrid(20, 1.0f);
                
                if (showSkeleton) {
                    DrawSkeleton(model, boneTransforms);
                }

                // Set the MVP matrix in the shader
                SetShaderValueMatrix(basicShader, mvpLoc, mvp);

                // Draw each mesh
                for (int i = 0; i < model.meshCount; i++) {
                    // Set the color based on the material
                    Vector4 color = {
                        model.materials[model.meshMaterial[i]].maps[MATERIAL_MAP_DIFFUSE].color.r / 255.0f,
                        model.materials[model.meshMaterial[i]].maps[MATERIAL_MAP_DIFFUSE].color.g / 255.0f,
                        model.materials[model.meshMaterial[i]].maps[MATERIAL_MAP_DIFFUSE].color.b / 255.0f,
                        model.materials[model.meshMaterial[i]].maps[MATERIAL_MAP_DIFFUSE].color.a / 255.0f
                    };
                    SetShaderValue(basicShader, colorLoc, &color, SHADER_UNIFORM_VEC4);
                    
                    DrawMesh(model.meshes[i], model.materials[model.meshMaterial[i]], modelMatrix);
                }

                // Draw coordinate system
                DrawLine3D((Vector3){0,0,0}, (Vector3){1,0,0}, RED);
                DrawLine3D((Vector3){0,0,0}, (Vector3){0,1,0}, GREEN);
                DrawLine3D((Vector3){0,0,0}, (Vector3){0,0,1}, BLUE);
            EndMode3D();

            // UI
            DrawFPS(10, 10);
            if (animationState.animationCount > 0) {
                DrawText(TextFormat("Animation: %s", animationState.animations[animationState.currentAnimation].name), 10, 40, 20, DARKGRAY);
                DrawText(TextFormat("Time: %.2f / %.2f", animationState.time, animationState.animations[animationState.currentAnimation].duration), 10, 70, 20, DARKGRAY);
            } else {
                DrawText("No Animations Available", 10, 40, 20, DARKGRAY);
            }
            DrawText("Press SPACE to Play/Pause", 10, 100, 20, DARKGRAY);
            DrawText("Press LEFT/RIGHT to Change Animation", 10, 130, 20, DARKGRAY);
            DrawText("Press D to Toggle Debug Info", 10, 160, 20, DARKGRAY);
            DrawText("Press S to Toggle Skeleton", 10, 190, 20, DARKGRAY);
            DrawText("Use +/- to Scale Model", 10, 220, 20, DARKGRAY);
            DrawText("Use WASD/QE to Move Model", 10, 250, 20, DARKGRAY);

            if (showDebugInfo) {
                DrawText("Debug Information:", 10, 280, 20, RED);
                DrawText(TextFormat("Model Scale: %.2f", modelScale), 10, 310, 20, RED);
                DrawText(TextFormat("Model Position: (%.2f, %.2f, %.2f)", modelPosition.x, modelPosition.y, modelPosition.z), 10, 340, 20, RED);
                DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", camera.position.x, camera.position.y, camera.position.z), 10, 370, 20, RED);
                DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", camera.target.x, camera.target.y, camera.target.z), 10, 400, 20, RED);
                // Additional debug info
                PrintDebugInfo(&animationState, &skeleton, nodes, nodeCount, boneTransforms);
            }

        EndDrawing();

    }

    printf("Cleaning up resources...\n");

    // Cleanup
    UnloadModel(model);
    for (int i = 0; i < textureCache.textureCount; ++i) {
        UnloadTexture(textureCache.textures[i]);
    }
    UnloadShader(basicShader);
    FreeAnimationData(&animationState, nodeCount);
    FreeGLTFData(data);
    free(myBoneInfos);
    free(inverseBindPose);
    free(boneTransforms);

    CloseWindow();

    printf("Application closed successfully.\n");
    return 0;
}

// Load the glTF file using cgltf
cgltf_data* LoadGLTFFile(const char* filename) {
	cgltf_options options = {0};
	cgltf_data* data = NULL;

	cgltf_result result = cgltf_parse_file(&options, filename, &data);
	if (result != cgltf_result_success) {
		printf("Failed to parse glTF file: %s\n", filename);
		return NULL;
	}

	result = cgltf_load_buffers(&options, data, filename);
	if (result != cgltf_result_success) {
		printf("Failed to load buffers for glTF file: %s\n", filename);
		cgltf_free(data);
		return NULL;
	}

	result = cgltf_validate(data);
	if (result != cgltf_result_success) {
		printf("Validation failed for glTF file: %s\n", filename);
		cgltf_free(data);
		return NULL;
	}

	printf("Successfully loaded glTF file: %s\n", filename);
	return data;
}

// Free the glTF data
void FreeGLTFData(cgltf_data* data) {
	cgltf_free(data);
}

// Load nodes into our Node structure
void LoadNodes(cgltf_data* data, Node* nodes, int* nodeCount) {
	for (cgltf_size i = 0; i < data->nodes_count; ++i) {
		cgltf_node* cgltfNode = &data->nodes[i];
		nodes[*nodeCount].name = cgltfNode->name ? cgltfNode->name : "Unnamed";
		nodes[*nodeCount].parentIndex = -1;
		nodes[*nodeCount].childCount = (int)cgltfNode->children_count;
		nodes[*nodeCount].hasAnimation = false;

		// Copy local transformation
		if (cgltfNode->has_matrix) {
			nodes[*nodeCount].transform = MatrixFromFloatArray((float*)cgltfNode->matrix);
		} else {
			nodes[*nodeCount].transform = MatrixIdentity();
			if (cgltfNode->has_translation) {
				nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
						 MatrixTranslate(cgltfNode->translation[0],
		       cgltfNode->translation[1],
		       cgltfNode->translation[2]));
			}
			if (cgltfNode->has_rotation) {
				Quaternion q = (Quaternion){cgltfNode->rotation[0], cgltfNode->rotation[1],
					cgltfNode->rotation[2], cgltfNode->rotation[3]};
				nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
						 QuaternionToMatrix(q));
			}
			if (cgltfNode->has_scale) {
				nodes[*nodeCount].transform = MatrixMultiply(nodes[*nodeCount].transform,
						 MatrixScale(cgltfNode->scale[0],
		   cgltfNode->scale[1],
		   cgltfNode->scale[2]));
			}
		}

		nodes[*nodeCount].globalTransform = nodes[*nodeCount].transform;

		(*nodeCount)++;
	}
}

// Build the node hierarchy
void BuildNodeHierarchy(Node* nodes, int nodeCount, cgltf_data* data) {
	for (int i = 0; i < nodeCount; ++i) {
		cgltf_node* cgltfNode = &data->nodes[i];

		// Set parent index
		if (cgltfNode->parent) {
			// Find parent index
			for (int j = 0; j < nodeCount; ++j) {
				if (&data->nodes[j] == cgltfNode->parent) {
					nodes[i].parentIndex = j;
					break;
				}
			}
		}

		// Set child indices
		for (cgltf_size k = 0; k < cgltfNode->children_count; ++k) {
			cgltf_node* childNode = cgltfNode->children[k];
			// Find child index
			for (int j = 0; j < nodeCount; ++j) {
				if (&data->nodes[j] == childNode) {
					nodes[i].childIndices[k] = j;
					break;
				}
			}
		}
	}
}

// Print node hierarchy recursively
void PrintNodeHierarchy(Node* nodes, int nodeIndex, int depth) {
	for (int i = 0; i < depth; ++i) printf("  ");
	printf("Node %d: %s\n", nodeIndex, nodes[nodeIndex].name);
	for (int i = 0; i < nodes[nodeIndex].childCount; ++i) {
		PrintNodeHierarchy(nodes, nodes[nodeIndex].childIndices[i], depth + 1);
	}
}

// Load skeleton (bones)
void LoadSkeleton(cgltf_data* data, Skeleton* skeleton, Node* nodes, int nodeCount) {
	if (data->skins_count == 0) {
		printf("No skeleton (skins) found in glTF file.\n");
		return;
	}

	cgltf_skin* skin = &data->skins[0];  // Assuming first skin
	skeleton->boneCount = (int)skin->joints_count;

	printf("\n--- Loading Skeleton with %d bones ---\n", skeleton->boneCount);

	for (int i = 0; i < skeleton->boneCount; ++i) {
		cgltf_node* jointNode = skin->joints[i];
		// Find the index of the jointNode in our nodes array
		int nodeIndex = -1;
		for (int j = 0; j < nodeCount; ++j) {
			if (&data->nodes[j] == jointNode) {
				nodeIndex = j;
				break;
			}
		}
		skeleton->bones[i].nodeIndex = nodeIndex;
		skeleton->bones[i].name = nodes[nodeIndex].name;

		printf("Bone %d: %s (Node Index: %d)\n", i, skeleton->bones[i].name, nodeIndex);
	}
}

// Load animations
void LoadAnimations(cgltf_data* data, AnimationState* animationState, Node* nodes, int nodeCount) {
	animationState->animationCount = (int)data->animations_count;
	printf("\n--- Loading %d animations ---\n", animationState->animationCount);

	for (int a = 0; a < animationState->animationCount; ++a) {
		cgltf_animation* animation = &data->animations[a];
		AnimationData* animData = &animationState->animations[a];
		animData->name = animation->name ? animation->name : "Unnamed Animation";
		animData->channelCount = (int)animation->channels_count;
		animData->duration = 0.0f;

		printf("Animation %d: %s\n", a, animData->name);

		// Initialize channels
		for (int i = 0; i < nodeCount; ++i) {
			animData->channels[i].keyframeCount = 0;
			animData->channels[i].times = NULL;
			animData->channels[i].translations = NULL;
			animData->channels[i].rotations = NULL;
			animData->channels[i].scales = NULL;
		}

		for (int i = 0; i < animData->channelCount; ++i) {
			cgltf_animation_channel* channel = &animation->channels[i];
			cgltf_animation_sampler* sampler = channel->sampler;

			// Find the index of the target node
			int nodeIndex = -1;
			for (int j = 0; j < nodeCount; ++j) {
				if (&data->nodes[j] == channel->target_node) {
					nodeIndex = j;
					nodes[j].hasAnimation = true;
					break;
				}
			}

			if (nodeIndex == -1) {
				printf("Warning: Animation channel target node not found.\n");
				continue;
			}

			AnimationChannel* animChannel = &animData->channels[nodeIndex];

			// Load input times
			cgltf_accessor* inputAccessor = sampler->input;
			size_t keyframeCount = inputAccessor->count;
			animChannel->keyframeCount = (int)keyframeCount;
			animChannel->times = malloc(sizeof(float) * keyframeCount);
			for (size_t k = 0; k < keyframeCount; ++k) {
				cgltf_accessor_read_float(inputAccessor, k, &animChannel->times[k], 1);
				if (animChannel->times[k] > animData->duration) {
					animData->duration = animChannel->times[k];
				}
			}

			// Load output values
			cgltf_accessor* outputAccessor = sampler->output;
			if (channel->target_path == cgltf_animation_path_type_translation) {
				animChannel->translations = malloc(sizeof(Vector3) * keyframeCount);
				for (size_t k = 0; k < keyframeCount; ++k) {
					float value[3];
					cgltf_accessor_read_float(outputAccessor, k, value, 3);
					animChannel->translations[k] = (Vector3){value[0], value[1], value[2]};
				}
			} else if (channel->target_path == cgltf_animation_path_type_rotation) {
				animChannel->rotations = malloc(sizeof(Quaternion) * keyframeCount);
				for (size_t k = 0; k < keyframeCount; ++k) {
					float value[4];
					cgltf_accessor_read_float(outputAccessor, k, value, 4);
					animChannel->rotations[k] = (Quaternion){value[0], value[1], value[2], value[3]};
				}
			} else if (channel->target_path == cgltf_animation_path_type_scale) {
				animChannel->scales = malloc(sizeof(Vector3) * keyframeCount);
				for (size_t k = 0; k < keyframeCount; ++k) {
					float value[3];
					cgltf_accessor_read_float(outputAccessor, k, value, 3);
					animChannel->scales[k] = (Vector3){value[0], value[1], value[2]};
				}
			}
		}
		printf("Animation '%s' duration: %f seconds\n", animData->name, animData->duration);
	}
}

// Load model (meshes) and materials
Model LoadModelFromGLTF(cgltf_data* data, Skeleton* skeleton, TextureCache* textureCache, Node* nodes, int nodeCount, MyBoneInfo** outMyBoneInfos, Matrix** outInverseBindPose) {
	printf("\n--- Loading meshes and materials from glTF file ---\n");
	Model model = {0};
	model.meshCount = 0;
	model.meshes = NULL;
	model.materialCount = 0;
	model.materials = NULL;

	// Handle materials
	LoadGLTFMaterials(data, &model, textureCache);

	// Count total number of primitives
	int totalPrimitives = 0;
	for (cgltf_size m = 0; m < data->meshes_count; ++m) {
		cgltf_mesh* cgltfMesh = &data->meshes[m];
		totalPrimitives += (int)cgltfMesh->primitives_count;
	}

	model.meshCount = totalPrimitives;
	model.meshes = (Mesh*)malloc(sizeof(Mesh) * model.meshCount);
	model.meshMaterial = (int*)malloc(sizeof(int) * model.meshCount);

	int meshIndex = 0;

	// Load meshes
	for (cgltf_size m = 0; m < data->meshes_count; ++m) {
		cgltf_mesh* cgltfMesh = &data->meshes[m];

		for (cgltf_size p = 0; p < cgltfMesh->primitives_count; ++p) {
			cgltf_primitive* primitive = &cgltfMesh->primitives[p];
			Mesh* mesh = &model.meshes[meshIndex];
			memset(mesh, 0, sizeof(Mesh));

			// Load indices
			cgltf_accessor* indexAccessor = primitive->indices;
			if (indexAccessor) {
				mesh->vertexCount = (int)indexAccessor->count;
				mesh->triangleCount = mesh->vertexCount / 3;
				mesh->indices = (unsigned short*)malloc(sizeof(unsigned short) * mesh->vertexCount);
				for (size_t i = 0; i < indexAccessor->count; ++i) {
					cgltf_uint tempIndex;
					cgltf_accessor_read_uint(indexAccessor, i, &tempIndex, 1);
					mesh->indices[i] = (unsigned short)tempIndex;
				}
			} else {
				// Non-indexed geometry
				printf("Warning: Primitive without indices is not supported.\n");
				continue;
			}

			// Load attributes
			for (cgltf_size a = 0; a < primitive->attributes_count; ++a) {
				cgltf_attribute* attribute = &primitive->attributes[a];
				cgltf_accessor* accessor = attribute->data;

				if (attribute->type == cgltf_attribute_type_position) {
					mesh->vertices = (float*)malloc(sizeof(float) * 3 * mesh->vertexCount);
					for (size_t i = 0; i < accessor->count; ++i) {
						float value[3];
						cgltf_accessor_read_float(accessor, i, value, 3);
						mesh->vertices[i * 3 + 0] = value[0];
						mesh->vertices[i * 3 + 1] = value[1];
						mesh->vertices[i * 3 + 2] = value[2];
					}
				} else if (attribute->type == cgltf_attribute_type_normal) {
					mesh->normals = (float*)malloc(sizeof(float) * 3 * mesh->vertexCount);
					for (size_t i = 0; i < accessor->count; ++i) {
						float value[3];
						cgltf_accessor_read_float(accessor, i, value, 3);
						mesh->normals[i * 3 + 0] = value[0];
						mesh->normals[i * 3 + 1] = value[1];
						mesh->normals[i * 3 + 2] = value[2];
					}
				} else if (attribute->type == cgltf_attribute_type_texcoord) {
					mesh->texcoords = (float*)malloc(sizeof(float) * 2 * mesh->vertexCount);
					for (size_t i = 0; i < accessor->count; ++i) {
						float value[2];
						cgltf_accessor_read_float(accessor, i, value, 2);
						mesh->texcoords[i * 2 + 0] = value[0];
						mesh->texcoords[i * 2 + 1] = value[1];
					}
				} else if (attribute->type == cgltf_attribute_type_joints) {
					mesh->boneIds = (unsigned char*)malloc(sizeof(unsigned char) * 4 * mesh->vertexCount);
					for (size_t i = 0; i < accessor->count; ++i) {
						cgltf_uint value[4];
						cgltf_accessor_read_uint(accessor, i, value, 4);
						mesh->boneIds[i * 4 + 0] = (unsigned char)value[0];
						mesh->boneIds[i * 4 + 1] = (unsigned char)value[1];
						mesh->boneIds[i * 4 + 2] = (unsigned char)value[2];
						mesh->boneIds[i * 4 + 3] = (unsigned char)value[3];
					}
				} else if (attribute->type == cgltf_attribute_type_weights) {
					mesh->boneWeights = (float*)malloc(sizeof(float) * 4 * mesh->vertexCount);
					for (size_t i = 0; i < accessor->count; ++i) {
						float value[4];
						cgltf_accessor_read_float(accessor, i, value, 4);
						mesh->boneWeights[i * 4 + 0] = value[0];
						mesh->boneWeights[i * 4 + 1] = value[1];
						mesh->boneWeights[i * 4 + 2] = value[2];
						mesh->boneWeights[i * 4 + 3] = value[3];
					}
				}
			}

			// Assign material
			if (primitive->material) {
				// Find material index
				int materialIndex = -1;
				for (int m = 0; m < model.materialCount; ++m) {
					if (&data->materials[m] == primitive->material) {
						materialIndex = m;
						break;
					}
				}
				if (materialIndex == -1) {
					printf("Warning: Material not found.\n");
					materialIndex = 0;
				}
				model.meshMaterial[meshIndex] = materialIndex;
			} else {
				model.meshMaterial[meshIndex] = 0; // Default material
			}

			// Upload mesh to GPU
			UploadMesh(mesh, true);

			meshIndex++;
		}
	}

	// Initialize bones
	model.boneCount = skeleton->boneCount;
	model.bones = (BoneInfo*)malloc(sizeof(BoneInfo) * model.boneCount);
	model.bindPose = (Transform*)malloc(sizeof(Transform) * model.boneCount);

	// Allocate MyBoneInfo array
	*outMyBoneInfos = (MyBoneInfo*)malloc(sizeof(MyBoneInfo) * model.boneCount);
	MyBoneInfo* myBoneInfos = *outMyBoneInfos;

	// Allocate inverse bind pose array
	Matrix* inverseBindPose = (Matrix*)malloc(sizeof(Matrix) * model.boneCount);

	// Update global transforms for nodes in bind pose
	for (int i = 0; i < nodeCount; ++i) {
		UpdateGlobalTransforms(nodes, i);
	}

	for (int i = 0; i < model.boneCount; ++i) {
		int nodeIndex = skeleton->bones[i].nodeIndex;

		// Initialize BoneInfo for model.bones
		model.bones[i].parent = nodes[nodeIndex].parentIndex;
		strncpy(model.bones[i].name, skeleton->bones[i].name, sizeof(model.bones[i].name) - 1);
		model.bones[i].name[sizeof(model.bones[i].name) - 1] = '\0'; // Ensure null-termination

		// Initialize MyBoneInfo
		myBoneInfos[i].nodeIndex = nodeIndex;
		myBoneInfos[i].name = skeleton->bones[i].name;

		// Capture bind pose transforms
		Matrix globalTransform = nodes[nodeIndex].globalTransform;

		// Store the bind pose as a Transform
		model.bindPose[i] = (Transform){
			.translation = MatrixToTranslation(globalTransform),
			.rotation = QuaternionFromMatrix(globalTransform),
			.scale = MatrixToScale(globalTransform)
		};

		// Convert bindPose to Matrix and calculate inverse
		Matrix bindPoseMatrix = TransformToMatrix(model.bindPose[i]);
		inverseBindPose[i] = MatrixInvert(bindPoseMatrix);
	}

	// Output inverse bind pose
	*outInverseBindPose = inverseBindPose;

	return model;
}

// Load materials and textures
void LoadGLTFMaterials(cgltf_data* data, Model* model, TextureCache* textureCache) {
	model->materialCount = (int)data->materials_count;
	if (model->materialCount == 0) {
		// Use default material
		model->materialCount = 1;
		model->materials = (Material*)malloc(sizeof(Material));
		model->materials[0] = LoadMaterialDefault();
		return;
	}

	model->materials = (Material*)malloc(sizeof(Material) * model->materialCount);

	// Base path for texture files
	char basePath[256] = {0};
	strcpy(basePath, GetDirectoryPath(GetWorkingDirectory()));

	for (int i = 0; i < model->materialCount; ++i) {
		cgltf_material* cgltfMat = &data->materials[i];
		Material* mat = &model->materials[i];
		*mat = LoadMaterialDefault();

		// Set material properties
		if (cgltfMat->has_pbr_metallic_roughness) {
			cgltf_pbr_metallic_roughness* pbr = &cgltfMat->pbr_metallic_roughness;

			// Base color
			mat->maps[MATERIAL_MAP_ALBEDO].color = (Color){
				(unsigned char)(pbr->base_color_factor[0] * 255),
				(unsigned char)(pbr->base_color_factor[1] * 255),
				(unsigned char)(pbr->base_color_factor[2] * 255),
				(unsigned char)(pbr->base_color_factor[3] * 255)
			};

			// Base color texture
			if (pbr->base_color_texture.texture && pbr->base_color_texture.texture->image) {
				const char* uri = pbr->base_color_texture.texture->image->uri;
				if (uri) {
					Texture2D texture = LoadTextureFromImageUri(basePath, uri);
					mat->maps[MATERIAL_MAP_ALBEDO].texture = texture;
				}
			}
		}

		// Normal map
		if (cgltfMat->normal_texture.texture && cgltfMat->normal_texture.texture->image) {
			const char* uri = cgltfMat->normal_texture.texture->image->uri;
			if (uri) {
				Texture2D texture = LoadTextureFromImageUri(basePath, uri);
				mat->maps[MATERIAL_MAP_NORMAL].texture = texture;
			}
		}

		// Metallic-Roughness map
		if (cgltfMat->has_pbr_metallic_roughness && cgltfMat->pbr_metallic_roughness.metallic_roughness_texture.texture && cgltfMat->pbr_metallic_roughness.metallic_roughness_texture.texture->image) {
			const char* uri = cgltfMat->pbr_metallic_roughness.metallic_roughness_texture.texture->image->uri;
			if (uri) {
				Texture2D texture = LoadTextureFromImageUri(basePath, uri);
				mat->maps[MATERIAL_MAP_ROUGHNESS].texture = texture;
			}
		}

		// Occlusion map
		if (cgltfMat->occlusion_texture.texture && cgltfMat->occlusion_texture.texture->image) {
			const char* uri = cgltfMat->occlusion_texture.texture->image->uri;
			if (uri) {
				Texture2D texture = LoadTextureFromImageUri(basePath, uri);
				mat->maps[MATERIAL_MAP_OCCLUSION].texture = texture;
			}
		}

		// Emissive map
		if (cgltfMat->emissive_texture.texture && cgltfMat->emissive_texture.texture->image) {
			const char* uri = cgltfMat->emissive_texture.texture->image->uri;
			if (uri) {
				Texture2D texture = LoadTextureFromImageUri(basePath, uri);
				mat->maps[MATERIAL_MAP_EMISSION].texture = texture;
			}
		}
	}
}

// Load texture from image URI
Texture2D LoadTextureFromImageUri(const char* basePath, const char* uri) {
	char fullPath[512];
	snprintf(fullPath, sizeof(fullPath), "%s/%s", basePath, uri);

	printf("Loading texture: %s\n", fullPath);
	Image image = LoadImage(fullPath);
	Texture2D texture = LoadTextureFromImage(image);
	UnloadImage(image);
	return texture;
}

// Update animation
void UpdateAnimation(AnimationState* animationState, Node* nodes, int nodeCount) {
	if (animationState->animationCount == 0) return;

	AnimationData* animData = &animationState->animations[animationState->currentAnimation];
	float time = animationState->time;

	for (int i = 0; i < nodeCount; ++i) {
		if (!nodes[i].hasAnimation) continue;
		AnimationChannel* channel = &animData->channels[i];
		if (channel->keyframeCount == 0) continue;

		// Find the current keyframe
		int frame = 0;
		while (frame < channel->keyframeCount - 1 && channel->times[frame + 1] < time) {
			frame++;
		}

		// Calculate interpolation factor
		float t = 0.0f;
		if (frame < channel->keyframeCount - 1) {
			float startTime = channel->times[frame];
			float endTime = channel->times[frame + 1];
			t = (time - startTime) / (endTime - startTime);
		}

		// Interpolate transformation
		Matrix transform = MatrixIdentity();
		if (channel->translations) {
			Vector3 start = channel->translations[frame];
			Vector3 end = channel->translations[frame + 1];
			Vector3 translation = Vector3Lerp(start, end, t);

			transform = MatrixMultiply(transform, MatrixTranslate(translation.x, translation.y, translation.z));
		}
		if (channel->rotations) {
			Quaternion start = channel->rotations[frame];
			Quaternion end = channel->rotations[frame + 1];
			Quaternion rotation = QuaternionSlerp(start, end, t);

			transform = MatrixMultiply(transform, QuaternionToMatrix(rotation));
		}
		if (channel->scales) {
			Vector3 start = channel->scales[frame];
			Vector3 end = channel->scales[frame + 1];
			Vector3 scale = Vector3Lerp(start, end, t);

			transform = MatrixMultiply(transform, MatrixScale(scale.x, scale.y, scale.z));
		}

		nodes[i].transform = transform;
	}

	// Update global transforms
	for (int i = 0; i < nodeCount; ++i) {
		UpdateGlobalTransforms(nodes, i);
	}
}

// Apply node transforms to the model's bones
void ApplyNodeTransforms(Node* nodes, int nodeCount, Skeleton* skeleton, MyBoneInfo* myBoneInfos, Matrix* inverseBindPose, Matrix* boneTransforms) {
	for (int i = 0; i < skeleton->boneCount; ++i) {
		int nodeIndex = myBoneInfos[i].nodeIndex;
		Matrix globalTransform = nodes[nodeIndex].globalTransform;

		// Compute the final bone transform
		boneTransforms[i] = MatrixMultiply(inverseBindPose[i], globalTransform);
	}
}

// Update global transforms recursively
void UpdateGlobalTransforms(Node* nodes, int nodeIndex) {
	Node* node = &nodes[nodeIndex];
	if (node->parentIndex == -1) {
		node->globalTransform = node->transform;
	} else {
		node->globalTransform = MatrixMultiply(nodes[node->parentIndex].globalTransform, node->transform);
	}
}

// Free animation data
void FreeAnimationData(AnimationState* animationState, int nodeCount) {
	for (int a = 0; a < animationState->animationCount; ++a) {
		AnimationData* animData = &animationState->animations[a];
		for (int i = 0; i < nodeCount; ++i) {
			AnimationChannel* channel = &animData->channels[i];
			if (channel->times) free(channel->times);
			if (channel->translations) free(channel->translations);
			if (channel->rotations) free(channel->rotations);
			if (channel->scales) free(channel->scales);
		}
	}
}

// Print bone matrices
void PrintBoneMatrices(Matrix* boneTransforms, Skeleton* skeleton) {
	printf("\n--- Bone Matrices ---\n");
	for (int i = 0; i < skeleton->boneCount; ++i) {
		printf("Bone %d: %s\n", i, skeleton->bones[i].name);
		Matrix m = boneTransforms[i];
		printf("Matrix:\n");
		printf("%f %f %f %f\n", m.m0, m.m4, m.m8, m.m12);
		printf("%f %f %f %f\n", m.m1, m.m5, m.m9, m.m13);
		printf("%f %f %f %f\n", m.m2, m.m6, m.m10, m.m14);
		printf("%f %f %f %f\n", m.m3, m.m7, m.m11, m.m15);
	}
}

// Print debug information
void PrintDebugInfo(AnimationState* animationState, Skeleton* skeleton, Node* nodes, int nodeCount, Matrix* boneTransforms) {
	// Print bone matrices
	PrintBoneMatrices(boneTransforms, skeleton);

	// Additional debug information can be added here
}

// Helper functions to extract translation, rotation, and scale from a matrix
Vector3 MatrixToTranslation(Matrix m) {
	return (Vector3){ m.m12, m.m13, m.m14 };
}

Vector3 MatrixToScale(Matrix m) {
	return (Vector3){
		Vector3Length((Vector3){ m.m0, m.m1, m.m2 }),
		Vector3Length((Vector3){ m.m4, m.m5, m.m6 }),
		Vector3Length((Vector3){ m.m8, m.m9, m.m10 })
	};
}

/*
Quaternion QuaternionFromMatrix(Matrix m) {
    Quaternion q = {0};
    float trace = m.m0 + m.m5 + m.m10;
    if (trace > 0) {
	float s = sqrtf(trace + 1.0f) * 2.0f;
	q.w = 0.25f * s;
	q.x = (m.m6 - m.m9) / s;
	q.y = (m.m8 - m.m2) / s;
	q.z = (m.m1 - m.m4) / s;
    } else if ((m.m0 > m.m5) && (m.m0 > m.m10)) {
	float s = sqrtf(1.0f + m.m0 - m.m5 - m.m10) * 2.0f;
	q.w = (m.m6 - m.m9) / s;
	q.x = 0.25f * s;
	q.y = (m.m4 + m.m1) / s;
	q.z = (m.m8 + m.m2) / s;
    } else if (m.m5 > m.m10) {
	float s = sqrtf(1.0f + m.m5 - m.m0 - m.m10) * 2.0f;
	q.w = (m.m8 - m.m2) / s;
	q.x = (m.m4 + m.m1) / s;
	q.y = 0.25f * s;
	q.z = (m.m9 + m.m6) / s;
    } else {
	float s = sqrtf(1.0f + m.m10 - m.m0 - m.m5) * 2.0f;
	q.w = (m.m1 - m.m4) / s;
	q.x = (m.m8 + m.m2) / s;
	q.y = (m.m9 + m.m6) / s;
	q.z = 0.25f * s;
    }
    return q;
}
*/

Matrix TransformToMatrix(Transform t) {
	Matrix matTranslation = MatrixTranslate(t.translation.x, t.translation.y, t.translation.z);
	Matrix matRotation = QuaternionToMatrix(t.rotation);
	Matrix matScale = MatrixScale(t.scale.x, t.scale.y, t.scale.z);

	// Combine transformations
	return MatrixMultiply(MatrixMultiply(matTranslation, matRotation), matScale);
}

Matrix MatrixFromFloatArray(float* mat) {
	Matrix result = {0};
	result.m0  = mat[0];
	result.m4  = mat[1];
	result.m8  = mat[2];
	result.m12 = mat[3];
	result.m1  = mat[4];
	result.m5  = mat[5];
	result.m9  = mat[6];
	result.m13 = mat[7];
	result.m2  = mat[8];
	result.m6  = mat[9];
	result.m10 = mat[10];
	result.m14 = mat[11];
	result.m3  = mat[12];
	result.m7  = mat[13];
	result.m11 = mat[14];
	result.m15 = mat[15];
	return result;
}

void SetShaderValueMatrixArray(Shader shader, int locIndex, const Matrix* matrices, int count) {
	int actualCount = (count > MAX_BONES) ? MAX_BONES : count;
	float* matrixData = (float*)malloc(actualCount * 16 * sizeof(float));
	for (int i = 0; i < actualCount; i++) {
		memcpy(matrixData + i * 16, &matrices[i], 16 * sizeof(float));
	}
	SetShaderValueV(shader, locIndex, matrixData, SHADER_UNIFORM_FLOAT, actualCount * 16);
	free(matrixData);
}
void DrawSkeleton(Model model, Matrix* boneTransforms) {
	for (int i = 0; i < model.boneCount; i++) {
		if (model.bones[i].parent >= 0) {
			Vector3 start = Vector3Transform((Vector3){0,0,0}, boneTransforms[i]);
			Vector3 end = Vector3Transform((Vector3){0,0,0}, boneTransforms[model.bones[i].parent]);
			DrawLine3D(start, end, RED);
		}
	}
}

void PrintMeshData(Mesh mesh) {
	printf("Vertex count: %d\n", mesh.vertexCount);
	printf("Triangle count: %d\n", mesh.triangleCount);
	if (mesh.vertices) {
		printf("First 3 vertices:\n");
		for (int i = 0; i < 3 && i < mesh.vertexCount; i++) {
			printf("  (%f, %f, %f)\n", mesh.vertices[i*3], mesh.vertices[i*3+1], mesh.vertices[i*3+2]);
		}
	} else {
		printf("No vertex data\n");
	}
	if (mesh.texcoords) {
		printf("Has texture coordinates\n");
	} else {
		printf("No texture coordinates\n");
	}
	if (mesh.normals) {
		printf("Has normals\n");
	} else {
		printf("No normals\n");
	}
}

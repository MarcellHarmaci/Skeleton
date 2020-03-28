//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Harmaci Marcell
// Neptun : V7BH6J
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

class SiriusTriangle {
public:
	vec2 p1;
	vec2 p2;
	vec2 p3;

public:
	SiriusTriangle(float x1, float x2, float y1, float y2, float z1, float z2) {
		p1 = vec2(x1, x2);
		p2 = vec2(y1, y2);
		p3 = vec2(z1, z2);
	}

	SiriusTriangle(vec2 p1, vec2 p2, vec2 p3) {
		this->p1 = p1;
		this->p2 = p2;
		this->p3 = p3;
	}
};

// My global variables

// vertex and fragment shaders
GPUProgram gpuProgram;

// virtual world on the GPU
unsigned int vao1;
unsigned int vao2;

// vertex buffer objects
unsigned int vbo1;
unsigned int vbo2;

float* baseCircle = new float[104];
int sizeOfVertexArray = 0;
float* vertexArray = new float[sizeOfVertexArray];

std::vector<vec2> trianglePoints;
std::vector<SiriusTriangle> triangles;
int cntClicks = 0;
int cntCircles = 0;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Generate and bind vao1 and vbo1
	glGenVertexArrays(1, &vao1);
	glBindVertexArray(vao1);
	glGenBuffers(1, &vbo1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE,	   // two floats/attrib, not fixed-point
		0, NULL);				   // stride, offset: tightly packed
	
	// Generate and bind vao2 and vbo2
	glGenVertexArrays(2, &vao2);
	glBindVertexArray(vao2);
	glGenBuffers(2, &vbo2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2);
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	
	
	// Center coords of circle
	baseCircle[0] = 0.0f;
	baseCircle[1] = 0.0f;

	// Calculate vertices of circle
	for (int i = 1; i <= 50; i++) {
		double phi = 2 * M_PI * i / (double)50;
		baseCircle[2 * i] = (float) cos(phi);
		baseCircle[2 * i + 1] = (float) sin(phi);
	}
	// Extra point for GL_TRIANGLE_FAN to finish circle
	baseCircle[102] = baseCircle[2];
	baseCircle[103] = baseCircle[3];
	
	// Rebind vao1, so base circle stores there
	glBindVertexArray(vao1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);

	// Copy to GPU target
	glBufferData(GL_ARRAY_BUFFER,
		104 * sizeof(float),	// # of bytes
		baseCircle,				// address
		GL_STATIC_DRAW);		// we do not change later

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	// Set CIRCLE color to WHITE (1, 1, 1)
	int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(colorLocation, 1.0f, 1.0f, 1.0f); // 3 floats

	glBindVertexArray(vao1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo1);

	// Draw base circle
	glDrawArrays(
		GL_TRIANGLE_FAN,
		0, /*startIdx*/
		52 /*# Elements*/
	);

	// Set TRIANGLE color to RED (1, 0, 0)
	colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(colorLocation, 1.0f, 0.0f, 0.0f); // 3 floats
	
	glBindVertexArray(vao2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2);

	// Update vertices in VRAM
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeOfVertexArray * sizeof(float),  // # bytes
		vertexArray,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	// Draw triangles
	glDrawArrays(
		GL_TRIANGLES,
		0,
		sizeOfVertexArray / 2
	);
	

	// Set circles color to GREEN
	//colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
	//glUniform3f(colorLocation, 0.0f, 1.0f, 0.0f); // 3 floats
	//
	//// Draw circles
	//for (int i = 0; i < cntCircles; i++) {
	//	glDrawArrays(
	//		GL_LINE_STRIP,
	//		offsetCircles + 50 * i,
	//		offsetCircles + 50 * i + 50
	//	);
	//}
	//
	//printf("%d\n", sizeOfVertexArray);
	//for (int i = 0; i < sizeOfVertexArray / 2; i++) {
	//	if (i == 52)
	//		printf("Circle1 coords from now on\n");
	//	if (i == 52 + 100)
	//		printf("Circle2 coords from now on\n");
	//	if (i == 52 + 200)
	//		printf("Circle3 coords from now on\n");
	//	if (vertexArray[2 * i] == 0 && vertexArray[2 * i + 1] == 0)
	//		printf("ORIGO");
	//	
	//	printf("(%3.2f; %3.2f)\n", vertexArray[2 * i], vertexArray[2 * i + 1]);
	//}

	glutSwapBuffers(); // exchange buffers for double buffering
}

vec2 calcCenter(vec2 p1, vec2 p2) {
	// inverse of p1
	vec2 invP1 = vec2(
		p1.x / (p1.x * p1.x + p1.y * p1.y),
		p1.y / (p1.x * p1.x + p1.y * p1.y)
	);

	// normal 1 and 2
	vec2 n = vec2(
		p1.x - p2.x,
		p1.y - p2.y
	);
	vec2 m = vec2(
		p1.x - invP1.x,
		p1.y - invP1.y
	);

	// midpoint 1 and 2
	vec2 i = vec2(
		(p1.x + p2.x) / 2.0f,
		(p1.y + p2.y) / 2.0f
	);
	vec2 j = vec2(
		(p1.x + invP1.x) / 2.0f,
		(p1.y + invP1.y) / 2.0f
	);

	vec2 center = vec2();
	center.x = (n.y * dot(m, j) - m.y * dot(n, i)) / (n.y * m.x - n.x * m.y);
	center.y = (dot(n, i) - n.x *center.x) / n.y;

	return center;
}

void drawTriangleAt3() {
	// Dynamicly make space for new coords 
	int oldSize = sizeOfVertexArray;
	sizeOfVertexArray += 6; // Longer with 3 * 2 floats
	float* temp = new float[sizeOfVertexArray];

	// Copy previous base and triangle vertices
	for (int i = 0; i < oldSize; i++) {
		temp[i] = vertexArray[i];
	}

	// Swap pointers and delete outdated array
	float* oldArray = vertexArray;
	vertexArray = temp;
	delete[] oldArray;

	// Add the 3 new coordinates to the vertexArray (before circle vetices)
	vertexArray[oldSize + 0] = trianglePoints.at(trianglePoints.size() - 3).x;
	vertexArray[oldSize + 1] = trianglePoints.at(trianglePoints.size() - 3).y;
	vertexArray[oldSize + 2] = trianglePoints.at(trianglePoints.size() - 2).x;
	vertexArray[oldSize + 3] = trianglePoints.at(trianglePoints.size() - 2).y;
	vertexArray[oldSize + 4] = trianglePoints.at(trianglePoints.size() - 1).x;
	vertexArray[oldSize + 5] = trianglePoints.at(trianglePoints.size() - 1).y;

	// TODO - remove later, this is for transparency while debugging
	//for (int i = 0; i < sizeOfVertexArray / 2; i++) {
	//	printf("(%3.2f, %3.2f)\t", vertexArray[2 * i], vertexArray[2 * i + 1]);
	//}
}

/*
void drawCirclesAt3(SiriusTriangle triangle) {
	// Increase offset because of new triangle before in vertexArray
	offsetCircles += 3;
	cntCircles += 3;

	// Dynamicly make space for new coords 
	int oldSize = sizeOfVertexArray;
	sizeOfVertexArray += 300; // Longer with 3 * 50 * 2 floats
	float* temp = new float[sizeOfVertexArray];

	// Copy previous vertices
	for (int i = 0; i < oldSize; i++) {
		temp[i] = vertexArray[i];
	}

	// Swap pointers and delete outdated array
	float* oldArray = vertexArray;
	vertexArray = temp;
	delete[] oldArray;

	vec2 c1 = calcCenter(triangle.p1, triangle.p2);
	vec2 c2 = calcCenter(triangle.p2, triangle.p3);
	vec2 c3 = calcCenter(triangle.p3, triangle.p1);
	float r1 = length(triangle.p1 - c1);
	float r2 = length(triangle.p2 - c3);
	float r3 = length(triangle.p3 - c3);

	for (int i = 0; i < 50; i++) {
		double phi = 2 * M_PI * i / 50;
		vertexArray[2 * i + sizeOfVertexArray - 300] = c1.x + cos(phi) * r1;
		vertexArray[2 * i + sizeOfVertexArray - 299] = c1.y + sin(phi) * r1;
	}

	for (int i = 0; i < 50; i++) {
		double phi = 2 * M_PI * i / 50;
		vertexArray[2 * i + sizeOfVertexArray - 200] = c2.x + cos(phi) * r2;
		vertexArray[2 * i + sizeOfVertexArray - 199] = c2.y + sin(phi) * r2;
	}

	for (int i = 0; i < 50; i++) {
		double phi = 2 * M_PI * i / 50;
		vertexArray[2 * i + sizeOfVertexArray - 100] = c3.x + cos(phi) * r3;
		vertexArray[2 * i + sizeOfVertexArray -  99] = c3.y + sin(phi) * r3;
	}
}
*/

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		// Check if click was in the circle
		if (length(vec2(cX, cY)) > 1) {
			printf("Invalid point. Clink in the circle!\n");
			return;
		}

		trianglePoints.push_back(vec2(cX, cY));
		cntClicks++;
		
		// TODO remove printing
		printf("Mouse1 pressed at (%3.2f, %3.2f)\n", cX, cY);

		switch (cntClicks % 3)
		{
		case 0:
			triangles.push_back(
				SiriusTriangle(
					vec2(trianglePoints.at(trianglePoints.size() - 3)),
					vec2(trianglePoints.at(trianglePoints.size() - 2)),
					vec2(trianglePoints.at(trianglePoints.size() - 1))
				)
			);

			for (int i = 0; i < triangles.size(); i++) {
				printf("Points of triangle: {(%3.2f, %3.2f) (%3.2f, %3.2f) (%3.2f, %3.2f)}\n",
					triangles[i].p1.x, triangles[i].p1.y, triangles[i].p2.x, triangles[i].p2.y, triangles[i].p3.x, triangles[i].p3.y);
			}
			printf("\n");

			// My draw function
			drawTriangleAt3();
			//drawCirclesAt3(triangles.at(triangles.size() - 1));

			// Invalidate
			glutPostRedisplay();
			break;

		case 1:
			// Do nothing
			break;

		case 2:
			vec2 p1 = trianglePoints.at(trianglePoints.size() - 2);
			vec2 p2 = trianglePoints.at(trianglePoints.size() - 1);

			vec2 center = calcCenter(p1, p2);
			float radius = length(center - p1);



			// Double and triple check - remove later
			float radius2 = length(center - p2);
			float radius3 = length(center - vec2(p1.x / (p1.x * p1.x + p1.y * p1.y), p1.y / (p1.x * p1.x + p1.y * p1.y)));
			printf(
				"Center: (%3.2f, %3.2f)\nRadius1: %3.2f\nRadius2: %3.2f\nRadius3: %3.2f\n",
				center.x, center.y, radius, radius2, radius3
			);
			//printf(
			//	"These should be equal:\n1 + r^2 = %3.2f\n|C| = %3.2f\n\n",
			//	1 + (radius * radius), length(center)
			//);

			break;
		}
	}
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	//float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	//float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

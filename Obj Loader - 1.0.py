# -*- encoding:utf8 -*-
import numpy as np
import glMatrixTransformator as tr
import logging

logger = logging.getLogger("pyassimp")
gllogger = logging.getLogger("OpenGL")
note = logging.getLogger("App")
note.setLevel(level=logging.INFO)
gllogger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

import OpenGL

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import glfw

import pyassimp
from pyassimp.postprocess import *
from pyassimp.helper import *

class matrixSceneTool:

    camera_pos = 0
    view_matrix = 0
    direction = 0
    right = 0
    up = 0
    horizontal_angle = 0
    vertical_angle = 0


    def __init__(self, window_width, window_height):
        self.__init__()
        self.window_width = window_width
        self.window_height = window_height

    def __init__(self):
        self.direction = np.array([0.0, 0.0, -1.0]);
        self.right = np.array([1.0, 0.0, 0.0]);
        self.up = np.cross(self.right, self.direction )

        self.camera_pos = np.array([0.0,0.0,2.0])


        self.view_matrix = np.transpose(np.matrix([[self.right[0],self.up[0],-self.direction[0],0],
                                                    [self.right[1],self.up[1],-self.direction[1],0],
                                                    [self.right[2],self.up[2],-self.direction[2],0],
                                                    [-self.camera_pos[0],-self.camera_pos[1],-self.camera_pos[2],1]]))

        self.horizontal_angle = 0.0
        self.vertical_angle = .0
        self.window_width = 0
        self.window_height = 0

    def calculate_view_matrix(self):

        self.view_matrix = np.transpose(np.matrix([ [self.right[0],self.up[0],-self.direction[0],0],
                                                    [self.right[1],self.up[1],-self.direction[1],0],
                                                    [self.right[2],self.up[2],-self.direction[2],0],
                                                    [-self.camera_pos[0],-self.camera_pos[1],-self.camera_pos[2],1]]))

class timeHelper:
    def __init__(self):
        self.actual_time = 0
        self.previous_time = 0

    def GetTimeDelta(self):
        delta = self.actual_time - self.previous_time
        return delta

class cartesianHelper:
    def __init__(self):

        #for animation
        self.x_trans = 0.0
        self.y_trans = 0.0

        #for rotations
        self.radius = 0.5
        self.angle = 0

        #for mouse handling
        self.x_pos_prev = 0
        self.y_pos_prev = 0


global matrixScene
matrixScene = matrixSceneTool()

global time
time = timeHelper()

global cartesian
cartesian = cartesianHelper()


def fileToStr(fileName):
    with open(fileName, 'rb') as f:
        fileString =""
        note.info(("file: "+ f.name))
        for line in f:
            fileString += line + '\n'
    f.close()
    return fileString

def loadVertexFragmentFromFile(vertexSPath, fragmentSPath):
    return fileToStr(vertexSPath), fileToStr(fragmentSPath)

def loadTexture(texPath):
    image = open(texPath)

def showInfoAboutGl():
    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    version = glGetString(GL_VERSION)
    shading = glGetString(GL_SHADING_LANGUAGE_VERSION)
    print vendor
    print renderer
    print version
    print shading

def animate():

    np.matrix('1,0,0,0; 0,1,0,0; 0,0,1,0;0,0,0,1', np.float32)

    time.actual_time = glfw.get_time()

    delta = time.actual_time - time.prev_time
    x_trans = 0
    y_trans = 0

    if(delta > 0.01):
        time.prev_time=time.actual_time
        cartesian.angle+=1
        x_trans = cartesian.radius * np.cos(cartesian.angle*np.pi/180.0)
        y_trans = cartesian.radius * np.sin(cartesian.angle*np.pi/180.0)

    trans = tr.translate(x_trans,y_trans,0.0)
    trans = tr.rotate(0,cartesian.angle,0,trans,True)

    return trans

def WindowSizeCallback(window,  width,  height):

    window_width = width;
    window_height = height;

    aspect = float(window_width) / float(window_height);

    #set new matrixScene sizes
    matrixScene.window_height = window_height
    matrixScene.window_width = window_width

def key_callback(window,  key,  scancode,  action,  mods):

    if (key == glfw.KEY_ESCAPE and action == glfw.PRESS):

        glfw.set_window_should_close(window, GL_TRUE)
        return;

    move_speed = 6.0

    camera_pos = matrixScene.camera_pos
    direction = matrixScene.direction
    right = matrixScene.right

    if (key == glfw.KEY_W):
        camera_pos += direction * time.GetTimeDelta() * move_speed;
    if (key == glfw.KEY_S):
        camera_pos -= direction * time.GetTimeDelta() * move_speed;
    if (key == glfw.KEY_A):
        camera_pos -= right * time.GetTimeDelta() * move_speed;
    if (key == glfw.KEY_D):
        camera_pos += right * time.GetTimeDelta() * move_speed;

    #set new camera_pos and calculate matrix
    matrixScene.camera_pos = camera_pos
    matrixScene.calculate_view_matrix()

    note.info(matrixScene.camera_pos)



def cursor_callback(win, x_pos, y_pos):
    if(x_pos != cartesian.x_pos_prev and y_pos != cartesian.y_pos_prev):

        glfw.set_cursor_pos(win, window_width/2, window_height/2);
        glfw.set_input_mode(win, glfw.CURSOR , glfw.CURSOR_NORMAL)

        horizontal_angle = matrixScene.horizontal_angle
        vertical_angle = matrixScene.vertical_angle
        mouse_speed = 4

        horizontal_angle += mouse_speed * time.GetTimeDelta() * float(window_width / 2.0 - x_pos);
        vertical_angle += mouse_speed * time.GetTimeDelta() * float(window_height / 2.0 - y_pos);
        if (vertical_angle < -1.57):
            vertical_angle = -1.57;
        if (vertical_angle > 1.57):
            vertical_angle = 1.57;


        direction = np.array([np.cos(vertical_angle) * np.sin(horizontal_angle), np.sin(vertical_angle), np.cos(vertical_angle) * np.cos(horizontal_angle)])
        right = np.array([-np.cos(horizontal_angle), 0, np.sin(horizontal_angle)]);
        up = np.cross(right, direction )

        matrixScene.direction = direction
        matrixScene.right = right
        matrixScene.up = up
        matrixScene.horizontal_angle = horizontal_angle
        matrixScene.vertical_angle = vertical_angle

        matrixScene.calculate_view_matrix()

        cartesian.x_pos_prev = x_pos
        cartesian.y_pos_prev = y_pos

class Obj:
    texture_files = dict()
    shader = {}

    def __init__(self, model, vertexShaderPath, fragmentShaderPath):
        self.generateShader(vertexShaderPath, fragmentShaderPath)
        self.load_model(model)

    def generateShader(self, vertPath, fragPath):
        vertexShader,  fragmentShader   = loadVertexFragmentFromFile(vertPath, fragPath)

        vertex = shaders.compileShader(vertexShader, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(fragmentShader, GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vertex, fragment)

    def prepare_gl_buffers(self, mesh):

        stride = 32  # 8 * 4 bytes

        mesh.gl = {}

        # Fill the buffer for vertex and normals positions
        v = np.array(mesh.vertices, 'f')
        n = np.array(mesh.normals, 'f')

        if(np.max(v) > 1.0):
            v = v/np.amax(v)
            n = n/np.amax(n)

        tex = mesh.texturecoords[0][:,[0,1]]

        tex =  np.concatenate( ( mesh.texturecoords[0][:,[0]] , np.ones((tex.shape[0], 1), dtype=np.float) - mesh.texturecoords[0][:,[1]] ) , axis=1)

        t = np.array(tex, 'f')

        tupleStack = (v, n)
        stack = numpy.hstack(tupleStack)
        stack = numpy.concatenate((stack,t), axis=1)
        mesh.gl["vbo"] = vbo.VBO(stack)

        # Fill the buffer for vertex positions
        mesh.gl["faces"] = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["faces"])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     mesh.faces,
                     GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)



    def load_model(self, path, postprocess=aiProcessPreset_TargetRealtime_MaxQuality):
        logger.info("Loading model:" + path + "...")

        if postprocess:
            self.scene = pyassimp.load(path, postprocess)
        else:
            self.scene = pyassimp.load(path)
        logger.info("Done.")

        scene = self.scene
        # log some statistics
        logger.info("  meshes: %d" % len(scene.meshes))
        logger.info("  total faces: %d" % sum([len(mesh.faces) for mesh in scene.meshes]))
        logger.info("  materials: %d" % len(scene.materials))


        for index, mesh in enumerate(scene.meshes):
            self.prepare_gl_buffers(mesh)

        self.putTextures(scene.meshes, scene.materials)

        pyassimp.release(scene)

        logger.info("Ready for 3D rendering!")


    def putTextures(self, meshes, materials):

        import PIL.Image

        for index, mesh in enumerate(meshes):

            material = materials[mesh.materialindex]

            if ( ('file', 1L) in material.properties):

                file_name = material.properties[('file', 1L)]

                if(not file_name in self.texture_files):
                    size = len(self.texture_files)

                    tex_no = 0x84C0 + size

                    im = PIL.Image.open(material.properties[('file', 1L)])
                    colors = im.mode
                    ix, iy = im.size[0], im.size[1]
                    image = im.tobytes()



                    if(colors == 'RGBA'):
                        glActiveTexture(tex_no);
                        texture = glGenTextures(1);
                        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
                        self.texture_files[file_name] =  (0x84C0 + size, texture)
                        glBindTexture(GL_TEXTURE_2D, texture);

                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA , GL_UNSIGNED_BYTE, image);

                        glGenerateMipmap(GL_TEXTURE_2D);
                        logger.info('RGBA TEX')
                    else:
                        glActiveTexture(tex_no);
                        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
                        texture = glGenTextures(1);
                        self.texture_files[file_name] =  (0x84C0 + size, texture)
                        glBindTexture(GL_TEXTURE_2D, texture);

                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ix, iy, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

                        glGenerateMipmap(GL_TEXTURE_2D);
                        logger.info('RGB TEX')

    def setPerspectiveMatrix(self):
        #macierz projekcji

        P1= 0.1
        P2 = 500.0
        FOV = 90.0
        aspect = float(window_width)/float(window_height)

        D = np.tan(0.5*FOV/180.0* np.pi) * P1
        Sx = (2.0*P1)/(2*D*aspect)
        Sy = 1.0*P1/D
        Sz = -(P2+P1)/(P2-P1)
        Pz = -(2.0 * P2 * P1)/(P2 - P1)

        global perspective
        self.perspective = np.matrix([[Sx,0,0,0],
                                 [0,Sy,0,0],
                                 [0,0,Sz,-1.0],
                                 [0,0,Pz,0]])

    def setTransformMatrix(self):
        global x_trans
        global y_trans
        global transform_matrix
        x_trans = y_trans = 0.1
        transform_matrix = np.matrix('1,0,0,0; 0,1,0,0; 0,0,1,0;0,0,0,1', np.float32)

    def enableDepth(self):
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

    def setTexFiltering(self):
        #najbliższym sąsiadem

        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        #liniowe
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        #trójliniowe
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        #anizotrpowe
        from OpenGL.GL.EXT import texture_filter_anisotropic
        anisotropy_factor = 0.0
        anisotropy_factor = glGetFloatv(texture_filter_anisotropic.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT);
        glTexParameterf(GL_TEXTURE_2D, texture_filter_anisotropic.GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy_factor);


    def render(self, wireframe=False):
        shader = self.shader

        glUseProgram(app.shader)
        self.enableDepth()
        self.setPerspectiveMatrix()
        self.setTransformMatrix()

        Light_specular = glGetUniformLocation(app.shader,"Light_specular")
        Global_ambient = glGetUniformLocation(app.shader,"Global_ambient")
        Light_ambient = glGetUniformLocation(app.shader,"Light_ambient")
        Light_diffuse = glGetUniformLocation(app.shader,"Light_diffuse")
        Light_location = glGetUniformLocation(app.shader,"Light_location")

        view_uniform = glGetUniformLocation(app.shader ,"view_matrix")
        perspective_uniform = glGetUniformLocation(app.shader,"perspective_matrix")
        trans_uniform = glGetUniformLocation(app.shader,'trans_matrix')

        glUniformMatrix4fv(trans_uniform, 1, GL_FALSE, transform_matrix);
        glUniformMatrix4fv(view_uniform, 1 ,GL_TRUE,matrixScene.view_matrix)
        glUniformMatrix4fv(perspective_uniform,1,GL_FALSE,self.perspective)


        glUniform4f(Light_specular, 1.0, 1.0, 1.0, 1)
        glUniform4f(Global_ambient, .4,.2,.2,.1 )
        glUniform4f(Light_ambient, 1.0, 1.0, 1.0, 1 )
        glUniform4f(Light_diffuse, 1.0, 1.0, 1.0 ,1 )
        glUniform3f(Light_location, 0.0, 3.0, 0.0 )

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)

        self.recursive_render(self.scene.rootnode, shader)

    def recursive_render(self, node, shader):
        for mesh in node.meshes:

            stride = 32

            diffuse = mesh.material.properties["diffuse"]
            if len(diffuse) == 3: diffuse.append(1.0)
            ambient = mesh.material.properties["ambient"]
            if len(ambient) == 3: ambient.append(1.0)
            specular = mesh.material.properties["specular"]
            if len(specular) == 3: specular.append(1.0)
            shininess = mesh.material.properties["shininess"]


            Material_diffuse = glGetUniformLocation(shader,"Material_diffuse")
            Material_ambient = glGetUniformLocation(shader,"Material_ambient")
            Material_specular = glGetUniformLocation(shader,"Material_specular")
            Material_shininess = glGetUniformLocation(shader, "shininess")

            glUniform4f(Material_diffuse, *diffuse)
            glUniform4f(Material_ambient, *ambient)
            glUniform4f(Material_specular, *specular)
            glUniform1f(Material_shininess, shininess)


            vbo = mesh.gl["vbo"]
            vbo.bind()
            glEnableVertexAttribArray(1);
            glEnableVertexAttribArray(2);
            glEnableVertexAttribArray(3);

            glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, vbo )
            glVertexAttribPointer(2, 3, GL_FLOAT, False, stride, vbo + 12 )
            glVertexAttribPointer(3, 2, GL_FLOAT, False, stride, vbo + 24 )

            if(('file', 1L) in mesh.material.properties):
                file_name = mesh.material.properties[('file', 1L)]
                texture_gl_id = (self.texture_files[file_name])[1]
                texture_slot = glGetUniformLocation(shader, "texture");

                glBindTexture(GL_TEXTURE_2D, texture_gl_id)
                glUniform1i(texture_slot, texture_gl_id-1)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["faces"])
            glDrawElements(GL_TRIANGLES, len(mesh.faces) * 3, GL_UNSIGNED_INT, None)

            vbo.unbind();
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(2)
            glDisableVertexAttribArray(3)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)


        for child in node.children:
            self.recursive_render(child, shader)

def setGLFWOptions(win):
    glfw.make_context_current(win)
    glfw.window_hint(glfw.SAMPLES,4)
    glfw.set_key_callback(win, key_callback);
    glfw.set_cursor_pos_callback(win, cursor_callback);
    glfw.set_window_size_callback(win, WindowSizeCallback);

def clearColorSetViewport(r,g,b,a=1.0):
        glClearColor(r, g, b, a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, window_width, window_height);

if __name__=='__main__':
    glfw.init()
    window_width = 640
    window_height = 480
    win = glfw.create_window(window_width , window_height,"GL", None, None)
    setGLFWOptions(win)

    matrixScene.window_width = window_width
    matrixScene.window_height = window_height

    time.actual_time = time.previous_time = glfw.get_time()

    showInfoAboutGl()


    app = Obj('./models/Cat.obj', 'shaders/13_vertex_shader.glsl', 'shaders/13_fragment_shader.glsl')
    app.setTexFiltering()

    while not glfw.window_should_close(win):
        time.previous_time = time.actual_time;
        time.actual_time = glfw.get_time();
        clearColorSetViewport(0.5,0.1,0.2)
        app.render()
        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.destroy_window(win)
    glfw.terminate()
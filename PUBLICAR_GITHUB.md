# Publicar en GitHub Pages

El repositorio local ya está preparado en la rama `main`.

## 1. Crear repositorio público

En GitHub, crear un repositorio público, por ejemplo:

`proyecto-oro-ml-mejorado`

No agregar README, `.gitignore` ni licencia desde GitHub porque ya existen archivos locales.

## 2. Conectar este proyecto con GitHub

Desde esta carpeta, ejecutar:

```bash
git remote add origin https://github.com/TU_USUARIO/proyecto-oro-ml-mejorado.git
git push -u origin main
```

## 3. Activar GitHub Pages

En GitHub:

1. Abrir el repositorio.
2. Entrar a `Settings`.
3. Entrar a `Pages`.
4. En `Build and deployment`, seleccionar `Deploy from a branch`.
5. Seleccionar rama `main` y carpeta `/root`.
6. Guardar.

## 4. Enlace público

Después de unos minutos, el enlace será:

`https://TU_USUARIO.github.io/proyecto-oro-ml-mejorado/`

Ese enlace abre directamente el entregable ubicado en `dashboard/index.html`.

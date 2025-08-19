document.getElementById('cargar').addEventListener('click', function() {
    const producto = document.getElementById('producto').value;
    const tipo = document.getElementById('tipo').value;

    const archivo = `plots/${producto}_${tipo}.json`;

    fetch(archivo)
        .then(response => {
            if (!response.ok) throw new Error('Archivo no encontrado');
            return response.json();
        })
        .then(data => {
            Plotly.newPlot('grafico', data.data, data.layout);
        })
        .catch(error => {
            alert("No se pudo cargar el gráfico: " + error);
        });
});


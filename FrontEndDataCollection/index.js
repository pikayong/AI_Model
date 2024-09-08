// const { Sequelize, DataTypes, Op } = require('sequelize');

const http = require('http');
const express = require('express');
// const _models = require('./models/_models')
// const _students = require('./models/Students')
// const _lecturers = require('./models/Lecturers')
// const _global = require('./global')
const path = require('path')
const fs = require('fs');


const app = express ();
app.use(express.json());

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log("Server Listening on PORT:", PORT)
});

app.set("views", path.join(__dirname, "views"))
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.use(express.static(__dirname + '/assets'));
const folderPath = 'D:\\My Projects\\Personal Projects\\AI_Model\\Data\\self\\dataset\\raw\\'

async function documentWrite(_fileName, _content) {
    try {
      const content = _content;
      await fs.writeFileSync(folderPath + _fileName, content);
    } catch (err) {
      console.log(err);
    }
  }


app.get('/', (request, response) => {

    response.render("index")
})

app.post('/postdata', (request, response) => {
    documentWrite(request.body.fileName, request.body.content)
    // response.send(request);
    return;
})
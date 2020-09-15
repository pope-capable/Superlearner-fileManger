"use strict";

const express = require("express");
const socketIO = require("socket.io");
const bodyParser = require("body-parser");
const FolderRoutes = require("./api/server/routes/folderRoutes");
const FileRoutes = require("./api/server/routes/fileRoutes");
const ProcessRoutes = require("./api/server/routes/processRoutes");
const ModelRoutes = require("./api/server/routes/modelRoutes");
const SuperRoutes = require("./api/server/routes/superlearnerRoutes");
const NotificationRoutes = require("./api/server/routes/notificationRoutes");

// import express from 'express';
// import bodyParser from 'body-parser';
// import userRoutes from './api/server/routes/UserRoutes';
// import otpRoutes from './api/server/routes/otpRoutes';

// config.config();

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.use(function (req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Credentials", "false");
  res.header("Access-Control-Allow-Headers", "*");
  next();
});

const port = process.env.PORT || 8000;

app.use("/folders", FolderRoutes);
app.use("/file", FileRoutes);
app.use("/process", ProcessRoutes);
app.use("/model", ModelRoutes);
app.use("/super", SuperRoutes);
app.use("/notifications", NotificationRoutes);

// when a random route is inputed
app.get("*", (req, res) =>
  res.status(200).send({
    message: "Main SuperLearner Folder Micro-service is Running.",
  })
);

var server = app.listen(port, () => {
  console.log(`Server is running on PORT ${port}`);
});

const io = socketIO(server);

const socketHandshake = require("./api/server/utils/socket")(io);

app.set("io", io);

module.exports = server;

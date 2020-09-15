const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class FolderServices {
  // Create project Record
  static async createFolder(data) {
    try {
      return await database.folders_mains.create(data);
    } catch (error) {
      throw error;
    }
  }

  // bulk create system folders
  static async createSystemFolders(data) {
    try {
      return await database.folders_mains.bulkCreate(data, { returning: true });
    } catch (error) {
      throw error;
    }
  }

  // get one project breakdown
  static async getProjectFolders(data) {
    try {
      return await database.folders_mains.findAll({
        where: { projectId: data.projectId },
      });
    } catch (error) {
      throw error;
    }
  }

  // get upload folder
  static async getProjectUploadFolder(data) {
    try {
      return await database.folders_mains.findOne({
        where: { projectId: data.projectId, name: "Uploads" },
      });
    } catch (error) {
      throw error;
    }
  }
  
  // get result folder
  static async getProjectResultFolder(data) {
    try {
      return await database.folders_mains.findOne({
        where: { projectId: data.projectId, name: "Results" },
      });
    } catch (error) {
      throw error;
    }
  }

  // get data-pp folder
  static async getProjectDPPFolder(data) {
    try {
      return await database.folders_mains.findOne({
        where: { projectId: data.projectId, name: "Pre-processed files" },
      });
    } catch (error) {
      throw error;
    }
  }
}

module.exports = FolderServices;

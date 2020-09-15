const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class FileServices {
  // Create project Record
  static async createFile(data) {
    try {
      return await database.files_mains.create(data);
    } catch (error) {
      throw error;
    }
  }

  // bulk create system folders
  static async createMultipleFiles(data) {
    try {
      return await database.files_mains.bulkCreate(data, { returning: true });
    } catch (error) {
      throw error;
    }
  }

  // get one project breakdown
  static async getFiles(data) {
    try {
      return await database.files_mains.findAll({
        where: { folderId: data.folderId },
      });
    } catch (error) {
      throw error;
    }
  }

//   get recent files 
static async getRecentFiles(data) {
    var today = new Date
    try {
      return await database.folders_mains.findAll({
        where: { projectId: data.projectId, name: "Results", include: {model: database.files_mains}}
      });
    } catch (error) {
      throw error;
    }
  }

    //   delete one file
    static async removeFile(data) {
        try {
            return await database.files_mains.destroy({
              where: { id: data.fileId },
            });
          } catch (error) {
            throw error;
          }
    }


}

module.exports = FileServices;

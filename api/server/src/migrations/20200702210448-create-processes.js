'use strict';
module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('processes', {
      id: {
        type: Sequelize.UUID,
        primaryKey: true,
        defaultValue: Sequelize.UUIDV1,
      },
      projectId: {
        type: Sequelize.UUID,
        allowNull: false,
        unique: false,
      },
      name: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false,
      },
      type: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false
      },
      status: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: false,
        defaultValue: "Running..."
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });
  },
  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('processes');
  }
};